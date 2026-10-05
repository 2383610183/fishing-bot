# -*- coding: utf-8 -*-
"""
三角洲行动 · 自动钓鱼（整合 GUI 版）

依赖：
    pip install keyboard soundcard numpy scipy
    脱钩检测（可选）需要额外两个库：
    pip install mss opencv-python

按 F8 或界面上的【停止】随时停。
校准提示区：点界面上「校准提示区」按钮，3 秒后在全屏截图上框选「刺鱼」提示文字。
"""

import base64
import ctypes
import json
import queue
import random
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import keyboard
import numpy as np
import soundcard as sc
from scipy.io import wavfile
from scipy.signal import correlate, resample_poly

try:  # 脱钩检测用的视觉库，没装就只禁用这一个功能
    import cv2
    import mss
    HAS_VISION = True
except ImportError:
    HAS_VISION = False

SAMPLE_RATE = 48_000
BLOCK_SIZE = 1_024

# 打包成 exe 后，默认模板找 exe 同目录；源码运行时找脚本同目录
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

HUD_TPL_PATH = BASE_DIR / "hud_template.png"      # 校准截下的提示区模板
HUD_REGION_PATH = BASE_DIR / "hud_region.json"    # 提示区屏幕坐标
SPLASH_REGION_PATH = BASE_DIR / "splash_region.json"  # 水花检测区屏幕坐标

# Windows 高 DPI 缩放下让 tkinter 用物理像素，保证框选坐标和截图坐标一致
if sys.platform == "win32":
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# ---------------- 鼠标右键按住（SendInput，缩放聚焦浮漂） ----------------
_u32 = ctypes.windll.user32 if sys.platform == "win32" else None


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("mi", _MOUSEINPUT)]


def _mouse_flag(flag):
    inp = _INPUT(0, _MOUSEINPUT(0, 0, 0, flag, 0, None))
    _u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


_rmb_held = False


def rmb_down():
    global _rmb_held
    if not _rmb_held and _u32:
        _mouse_flag(0x0008)  # RIGHTDOWN
        _rmb_held = True


def rmb_up():
    global _rmb_held
    if _rmb_held and _u32:
        _mouse_flag(0x0010)  # RIGHTUP
        _rmb_held = False




# ============================================================
# 音频模板与匹配（来自 fishing_bot.py，逻辑未改）
# ============================================================

def load_template(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到声音模板：{path}")
    rate, audio = wavfile.read(path)
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        max_value = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)
    if rate != SAMPLE_RATE:
        divisor = np.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)
    audio -= np.mean(audio)
    norm = float(np.linalg.norm(audio))
    if norm < 1e-8:
        raise RuntimeError(f"模板是静音，无法使用：{path.name}")
    return audio, norm


def to_mono_float(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.nan_to_num(audio)


def match_score(audio, template, template_norm):
    """返回音频窗口中与模板最高的归一化相似度 (-1 ~ 1)。"""
    audio = np.asarray(audio, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    width = len(template)
    if len(audio) < width:
        return 0.0
    correlation = correlate(audio, template, mode="valid", method="fft")
    cumulative = np.concatenate(([0.0], np.cumsum(audio)))
    cumulative_sq = np.concatenate(([0.0], np.cumsum(audio * audio)))
    window_sum = cumulative[width:] - cumulative[:-width]
    window_sq_sum = cumulative_sq[width:] - cumulative_sq[:-width]
    window_energy = window_sq_sum - (window_sum * window_sum / width)
    denominator = np.sqrt(np.maximum(window_energy, 1e-12)) * template_norm
    scores = correlation / denominator
    return float(np.clip(np.max(scores), -1.0, 1.0))


# ============================================================
# 提示区模板匹配（脱钩检测）
# ============================================================

class HudWatcher:
    """比对提示区实时截图和校准模板。相似度高 = 还是「刺鱼」提示 = 线还在水里。"""

    def __init__(self, template_path, region_path):
        if not HAS_VISION:
            raise RuntimeError("缺少库，请先 pip install mss opencv-python")
        template_path = Path(template_path)
        region_path = Path(region_path)
        if not template_path.exists() or not region_path.exists():
            raise RuntimeError("还没校准提示区")
        self.tpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if self.tpl is None:
            raise RuntimeError("模板图片损坏，请重新校准")
        self.region = json.loads(region_path.read_text(encoding="utf-8"))
        self.sct = mss.mss()

    def similarity(self):
        img = np.asarray(self.sct.grab(self.region))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        if gray.shape != self.tpl.shape:
            return 0.0
        # 同尺寸匹配，结果是 1x1 的相似度（-1 ~ 1，越接近 1 越像）
        return float(cv2.matchTemplate(gray, self.tpl, cv2.TM_CCOEFF_NORMED)[0][0])


class SplashWatcher:
    """水花检测：盯一块水面区域，数「亮白色低饱和」像素。
    鱼咬钩瞬间水花炸开，白色像素暴涨；平时水波/浮漂假动作几乎不产生白色像素。"""

    def __init__(self, region_path):
        if not HAS_VISION:
            raise RuntimeError("缺少库，请先 pip install mss opencv-python")
        region_path = Path(region_path)
        if not region_path.exists():
            raise RuntimeError("还没框选水花区")
        self.region = json.loads(region_path.read_text(encoding="utf-8"))
        self.sct = mss.mss()

    def white_count(self):
        img = np.asarray(self.sct.grab(self.region))
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 150), (180, 100, 255))
        return cv2.countNonZero(mask)


# ============================================================
# 钓鱼机器人（后台线程）
# ============================================================

class FishingBot(threading.Thread):
    """
    通过 q 往 GUI 发消息：
      ("log", 文字)           日志
      ("score", 相似度)       实时相似度
      ("stats", 轮, 中, 空)   统计
      ("state", 状态文字)     当前阶段
      ("done",)               线程结束
    """

    def __init__(self, cfg, q):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.q = q
        self.stop_event = threading.Event()

    # ---------- 基础工具 ----------

    def log(self, msg):
        self.q.put(("log", msg))

    def state(self, msg):
        self.q.put(("state", msg))

    def stop_requested(self):
        if self.stop_event.is_set():
            return True
        try:
            if keyboard.is_pressed(self.cfg["stop_key"]):
                return True
        except Exception:
            pass
        return False

    def wait_interruptibly(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.stop_requested():
                return False
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        return not self.stop_requested()

    def press_action(self):
        key = self.cfg["action_key"]
        keyboard.press(key)
        try:
            time.sleep(self.cfg["hold_seconds"])
        finally:
            keyboard.release(key)

    # ---------- 听音 ----------

    def listen_for_bite(self, device, seconds, template, template_norm, hud=None,
                        splash=None):
        """监听咬钩声，返回 'bite' / 'bite_splash' / 'timeout' / 'escaped' / 'stopped'。"""
        chunks = []
        max_samples = len(template) + SAMPLE_RATE
        highest = 0.0
        audio_on = self.cfg["sound_detect"]
        deadline = time.monotonic() + seconds
        blocks_since_check = 0
        # 前 2 秒提示文字可能还没出现/在切换，不做模板判断
        next_hud_check = time.monotonic() + 2.0
        mismatch_since = None
        first_sim_logged = False
        splash_hits = 0
        splash_highest = 0
        first_sim_logged_splash = False
        prev_wc = 0
        t0 = time.monotonic()

        with device.recorder(samplerate=SAMPLE_RATE, channels=None,
                             blocksize=BLOCK_SIZE) as recorder:
            while time.monotonic() < deadline:
                if self.stop_requested():
                    return "stopped"
                chunks.append(to_mono_float(recorder.record(numframes=BLOCK_SIZE)))
                blocks_since_check += 1

                total = sum(len(c) for c in chunks)
                while len(chunks) > 1 and total - len(chunks[0]) >= max_samples:
                    total -= len(chunks[0])
                    chunks.pop(0)

                # 提示区模板匹配：每 0.4s 比一次，持续偏离才判定脱钩（防瞬间闪烁误报）
                if hud is not None and time.monotonic() >= next_hud_check:
                    next_hud_check = time.monotonic() + 0.4
                    try:
                        sim = hud.similarity()
                    except Exception as exc:
                        self.log(f"  脱钩检测出错，已自动关闭：{exc}")
                        hud = None
                        continue
                    if not first_sim_logged:
                        first_sim_logged = True
                        self.log(f"  入水后提示区相似度 {sim:.2f}"
                                 + ("（偏低！校准时线可能不在水里）"
                                    if sim < self.cfg["escape_threshold"] else ""))
                    if sim < self.cfg["escape_threshold"]:
                        mismatch_since = mismatch_since or time.monotonic()
                        if time.monotonic() - mismatch_since >= self.cfg["escape_grace"]:
                            self.log(f"  提示区变了（模板相似度 {sim:.2f}），"
                                     f"判定脱钩/线已不在水里")
                            return "escaped"
                    else:
                        mismatch_since = None

                # 水花检测：每个音频块查一次（约 20 毫秒）
                # 入水后 splash_mute 秒内不看（甩杆入水本身有大水花）
                if splash is not None and (
                        time.monotonic() - t0 >= self.cfg["splash_mute"]):
                    try:
                        wc = splash.white_count()
                    except Exception as exc:
                        self.log(f"  水花检测出错，已自动关闭：{exc}")
                        splash = None
                        wc = 0
                    if splash is not None:
                        splash_highest = max(splash_highest, wc)
                        if not first_sim_logged_splash:
                            first_sim_logged_splash = True
                            self.log(f"  入水后水花区白色像素 {wc}"
                                     + ("（暗场景等待期为 0 正常；若咬钩时也是 0，说明框选区域不对，请重新框选）"
                                        if wc == 0 else ""))
                        # 上升沿触发：白像素单次暴涨（光环开始扩散=咬钩瞬间），最快速度
                        if (wc >= self.cfg["splash_jump_min"]
                                and wc - prev_wc >= self.cfg["splash_jump"]):
                            self.log(f"  水花暴涨（{prev_wc} → {wc}）→ 判定咬钩")
                            return "bite_splash"
                        if wc >= self.cfg["splash_threshold"] * 2:
                            # 暴涨到 2 倍阈值以上：必是真水花，立即提竿
                            self.log(f"  看到大水花（白色像素 {wc}）→ 判定咬钩")
                            return "bite_splash"
                        if wc >= self.cfg["splash_threshold"]:
                            splash_hits += 1
                            if splash_hits >= 2:
                                self.log(f"  看到水花（白色像素 {wc}）→ 判定咬钩")
                                return "bite_splash"
                        else:
                            splash_hits = 0
                        prev_wc = wc

                if not audio_on or blocks_since_check < 6:
                    continue
                blocks_since_check = 0

                score = match_score(np.concatenate(chunks), template, template_norm)
                highest = max(highest, score)
                self.q.put(("score", score))

                if score >= self.cfg["threshold"]:
                    return "bite"
        extra = f"，水花白色像素峰值 {splash_highest}" if splash is not None else ""
        self.log(f"  超时未咬钩（最高相似度 {highest:.2f}{extra}）")
        return "timeout"

    def wait_animation(self):
        """收竿后干等动画结束（空竿或未开跳过时用）。返回 False 表示被停止。"""
        wait = random.uniform(self.cfg["anim_min"], self.cfg["anim_max"])
        self.state(f"等收竿动画 {wait:.1f}s")
        return self.wait_interruptibly(wait)

    # ---------- 主循环 ----------

    def run(self):
        try:
            self._run_inner()
        except Exception:
            import traceback
            self.log("发生未处理的错误，机器人已停止：")
            for line in traceback.format_exc().strip().splitlines():
                self.log("  " + line)
            self.q.put(("done",))

    def _run_inner(self):
        cfg = self.cfg
        try:
            bite_tpl = load_template(cfg["bite_template"])
        except Exception as exc:
            self.log(f"模板加载失败：{exc}")
            self.q.put(("done",))
            return

        # 打开设备 loopback
        try:
            device = sc.get_microphone(id=cfg["device_id"], include_loopback=True)
        except Exception as exc:
            self.log(f"无法打开设备环回：{exc}")
            self.q.put(("done",))
            return

        # 脱钩检测（可选）
        hud = None
        if cfg["escape_detect"]:
            try:
                hud = HudWatcher(HUD_TPL_PATH, HUD_REGION_PATH)
                self.log("脱钩检测已启用：提示区文字一变就重抛")
            except Exception as exc:
                self.log(f"脱钩检测未启用：{exc}")

        # 水花检测（可选，图像识别咬钩，不受旁人声音影响）
        splash = None
        if cfg["splash_detect"]:
            try:
                splash = SplashWatcher(SPLASH_REGION_PATH)
                base = splash.white_count()
                self.log(f"水花检测已启用：白色像素 ≥ {cfg['splash_threshold']:.0f} 判定咬钩，"
                         f"当前区域基准值 {base}")
            except Exception as exc:
                self.log(f"水花检测未启用：{exc}")

        self.log(f"监听设备：{cfg['device_name']}")
        self.log(f"咬钩模板：{Path(cfg['bite_template']).name}")
        self.log(f"请确认游戏已把抛竿/收竿绑到 {cfg['action_key'].upper()}，"
                 f"游戏窗口在最前面。倒计时 {cfg['countdown']} 秒后开始。")

        for remaining in range(cfg["countdown"], 0, -1):
            if self.stop_requested():
                break
            self.state(f"{remaining} 秒后开始……")
            time.sleep(1)

        cycle = caught = missed = fast_escapes = consec_timeout = 0
        while not self.stop_requested():
            cycle += 1
            self.q.put(("stats", (cycle, caught, missed)))
            self.log(f"—— 第 {cycle} 轮 ——")

            # 1) 抛竿（按两下，和脚本1一致）
            self.state("抛竿")
            self.press_action()
            if not self.wait_interruptibly(cfg["second_press_delay"]):
                break
            self.press_action()
            cast_time = time.monotonic()

            # 2) 抛竿 1 秒后按住右键缩放（浮漂聚焦到屏幕中心，水花检测更准）
            if cfg["hold_rmb"]:
                if not self.wait_interruptibly(1.0):
                    break
                rmb_down()

            # 3) 静音期，过滤抛竿噪音
            muted = cfg["ignore_seconds"] - (1.0 if cfg["hold_rmb"] else 0.0)
            if not self.wait_interruptibly(max(0.0, muted)):
                rmb_up()
                break

            # 4) 听咬钩 / 看水花
            listen_seconds = max(
                0.0, cfg["max_cast_seconds"] - (time.monotonic() - cast_time))
            self.state(f"监听咬钩（{listen_seconds:.0f}s 内）")
            listen_start = time.monotonic()
            result = self.listen_for_bite(device, listen_seconds, *bite_tpl,
                                          hud=hud, splash=splash)
            rmb_up()  # 上鱼/超时即松开右键
            if result == "stopped":
                break

            # 防呆：连续两轮入水几秒内就"脱钩"，基本是校准时线不在水里，自动关闭
            if hud is not None and result == "escaped":
                fast_escapes = (fast_escapes + 1
                                if time.monotonic() - listen_start < 5 else 0)
                if fast_escapes >= 2:
                    hud = None
                    self.log("连续两轮刚入水就判定脱钩，校准很可能是错的"
                             "（校准时线不在水里）。已自动关闭脱钩检测；"
                             "请重新校准：线抛进水里后点「校准提示区」框选提示文字。")
            elif result in ("bite", "bite_splash"):
                fast_escapes = 0

            # 5) 收竿
            if result == "timeout":
                consec_timeout += 1
                if (cfg["timeout_alert"] > 0
                        and consec_timeout >= cfg["timeout_alert"]):
                    consec_timeout = 0
                    self.q.put(("alert",
                                f"连续 {cfg['timeout_alert']} 轮没咬钩，"
                                f"可能没鱼饵了或不在正确钓鱼状态，请检查游戏画面。"))
            else:
                consec_timeout = 0
            if result in ("bite", "bite_splash"):
                if result == "bite_splash":
                    # 水花触发：白色像素出现=鱼已咬钩，立刻提竿，只留极小随机延迟
                    delay = random.uniform(0.03, 0.12)
                else:
                    delay = random.uniform(cfg["react_min"], cfg["react_max"])
                self.log(f"  咬钩！等了 {time.monotonic() - listen_start:.1f}s，"
                         f"随机再等 {delay:.3f}s 收竿")
                self.state("咬钩！准备收竿")
                if not self.wait_interruptibly(delay):
                    break
                caught += 1
            else:
                missed += 1
            self.press_action()
            self.q.put(("stats", (cycle, caught, missed)))

            # 5) 收竿后的处理
            if result in ("bite", "bite_splash") and cfg["skip_anim"]:
                # 钓到鱼：隔 X 秒按一次动作键跳过展示动画，马上接下一竿
                if not self.wait_interruptibly(cfg["skip_delay"]):
                    break
                self.state("跳过动画")
                self.press_action()
                self.log(f"  收竿 {cfg['skip_delay']:.1f}s 后按 {cfg['action_key'].upper()} 跳过动画")
                if not self.wait_interruptibly(cfg["post_skip_delay"]):
                    break
            else:
                # 空竿：默认也按键跳过收线动画（autofish 的做法，2~3 秒就能重抛）
                if cfg["skip_anim_miss"]:
                    self.state("空竿，跳过收线动画")
                    if not self.wait_interruptibly(cfg["miss_skip_delay"]):
                        break
                    self.press_action()
                    self.log(f"  空竿：收竿 {cfg['miss_skip_delay']:.1f}s 后按键跳过动画")
                    if not self.wait_interruptibly(cfg["post_skip_delay"]):
                        break
                else:
                    if not self.wait_animation():
                        break

            # 6) 每 N 轮停下来等上饵（bait_every=0 关闭）
            if cfg["bait_every"] > 0 and cycle % cfg["bait_every"] == 0:
                self.log(f"  已钓 {cycle} 轮，等 {cfg['rebait_delay']:.1f}s 上饵")
                self.state(f"上饵中（{cfg['rebait_delay']:.1f}s）")
                if not self.wait_interruptibly(cfg["rebait_delay"]):
                    break

        rmb_up()  # 兜底：防止停止时右键还按着
        self.log(f"已停止。共 {cycle} 轮，咬钩 {caught} 次，空竿 {missed} 次。")
        self.state("已停止")
        self.q.put(("score", 0.0))
        self.q.put(("done",))




# ============================================================
# GUI（极简版）
# ============================================================

# ---------------- 极简配色与字体 ----------------
C_BG     = "#faf9f7"   # 米白底
C_TEXT   = "#1c1b1a"   # 近黑文字
C_MUTED  = "#9b968b"   # 灰棕辅助文字
C_LINE   = "#e8e4dc"   # 细分隔线
C_FIELD  = "#ffffff"   # 输入框底
C_DARK   = "#1c1b1a"   # 主按钮（近黑）
C_ACCENT = "#2f6b4f"   # 深绿点缀
C_LOG_BG = "#171713"   # 日志深底
C_LOG_FG = "#cfccc0"   # 日志文字

F_TITLE = ("Microsoft YaHei UI Light", 19)
F_SUB   = ("Microsoft YaHei UI", 8)
F_CAP   = ("Microsoft YaHei UI", 8)          # 分区小标题
F_BODY  = ("Microsoft YaHei UI", 9)
F_STATE = ("Microsoft YaHei UI", 11, "bold")
F_BTN   = ("Microsoft YaHei UI", 11, "bold")
F_LOG   = ("Consolas", 9)

CONFIG_PATH = BASE_DIR / "fishing_gui_config.json"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("自动钓鱼")
        self.configure(bg=C_BG)
        self.q = queue.Queue()
        self.bot = None
        self._params_visible = False

        self._setup_style()

        # 底部按钮先 pack（side=bottom），保证任何窗口高度下都看得见
        self._build_buttons()
        self._build_header()

        content = ttk.Frame(self)
        content.pack(fill="both", expand=True, padx=22, pady=(4, 0))
        self._build_device_section(content)
        self._build_template_section(content)
        self._build_escape_section(content)
        self._build_param_section(content)
        self._build_status_section(content)
        self._build_log_section(content)

        self._load_config()

        # 窗口高度取内容实际需要的高度，打开即完整，不用手拉
        self.update_idletasks()
        w = 640
        h = min(self.winfo_reqheight() + 16, self.winfo_screenheight() - 90)
        self.geometry(f"{w}x{h}")
        self.minsize(600, 540)

        self.after(100, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- 样式 ----------

    def _setup_style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=C_BG, foreground=C_TEXT, font=F_BODY)
        st.configure("TFrame", background=C_BG)
        st.configure("TLabel", background=C_BG, foreground=C_TEXT)
        st.configure("Muted.TLabel", background=C_BG, foreground=C_MUTED, font=F_SUB)
        st.configure("Cap.TLabel", background=C_BG, foreground=C_MUTED, font=F_CAP)
        st.configure("State.TLabel", background=C_BG, foreground=C_ACCENT, font=F_STATE)
        st.configure("TCheckbutton", background=C_BG, foreground=C_TEXT)
        st.map("TCheckbutton", background=[("active", C_BG)])
        st.configure("TCombobox", padding=4, fieldbackground=C_FIELD)
        st.configure("Accent.Horizontal.TProgressbar",
                     troughcolor=C_LINE, background=C_ACCENT,
                     bordercolor=C_BG, lightcolor=C_ACCENT, darkcolor=C_ACCENT)

    def _section(self, parent, title):
        """细分隔线 + 灰棕小标题，极简分区。"""
        tk.Frame(parent, bg=C_LINE, height=1).pack(fill="x", pady=(12, 0))
        ttk.Label(parent, text=title, style="Cap.TLabel").pack(anchor="w", pady=(6, 4))
        inner = ttk.Frame(parent)
        inner.pack(fill="x")
        return inner

    def _entry(self, parent, var, width=None):
        e = tk.Entry(parent, textvariable=var, relief="flat", bg=C_FIELD,
                     fg=C_TEXT, font=F_BODY, highlightthickness=1,
                     highlightbackground=C_LINE, highlightcolor=C_ACCENT,
                     insertbackground=C_TEXT)
        if width:
            e.config(width=width)
        return e

    def _flat_button(self, parent, text, command, dark=False):
        return tk.Button(
            parent, text=text, command=command, relief="flat", cursor="hand2",
            font=F_BODY, padx=12, pady=3,
            bg=C_DARK if dark else C_FIELD,
            fg="white" if dark else C_TEXT,
            activebackground=C_DARK if dark else C_LINE,
            activeforeground="white" if dark else C_TEXT,
            highlightthickness=1, highlightbackground=C_LINE)

    def _build_header(self):
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", padx=22, pady=(16, 0))
        ttk.Label(hdr, text="自动钓鱼", font=F_TITLE).pack(side="left")
        ttk.Label(hdr, text="  Delta Force · 声音检测",
                  style="Muted.TLabel").pack(side="left", pady=(10, 0))
        ttk.Label(hdr, text="小小秦 · F8 停止",
                  style="Muted.TLabel").pack(side="right", pady=(10, 0))

    # ---------- 各分区 ----------

    def _build_device_section(self, parent):
        inner = self._section(parent, "声音设备")
        self.speakers = []
        try:
            self.speakers = sc.all_speakers()
            default = sc.default_speaker()
            default_name = default.name
        except Exception:
            default_name = ""
        self.device_var = tk.StringVar(value=default_name)
        names = [s.name for s in self.speakers]
        self.device_combo = ttk.Combobox(
            inner, textvariable=self.device_var, values=names, state="readonly")
        self.device_combo.pack(fill="x")
        if not self.speakers:
            ttk.Label(
                inner,
                text="⚠ 检测不到播放设备：远程桌面请开启声音重定向，"
                     "或检查 Windows Audio 服务和声卡驱动",
                foreground="#b3554d", font=F_SUB,
                wraplength=560).pack(anchor="w", pady=(6, 0))

    def _build_template_section(self, parent):
        inner = self._section(parent, "声音模板")
        default_bite = BASE_DIR / "bite_clean_10.5_to_12.wav"
        self.bite_var = tk.StringVar(value=str(default_bite))
        self._file_row(inner, "咬钩音效", self.bite_var)

    def _file_row(self, parent, label, var):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=16,
                  style="Muted.TLabel").pack(side="left")
        self._entry(row, var).pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._flat_button(row, "浏览", lambda: self._browse(var)).pack(side="left")

    def _browse(self, var):
        path = filedialog.askopenfilename(
            filetypes=[("WAV 音频", "*.wav"), ("所有文件", "*.*")])
        if path:
            var.set(path)

    # ---------- 脱钩检测 / 校准 ----------

    def _build_escape_section(self, parent):
        inner = self._section(parent, "视觉检测")
        row = ttk.Frame(inner)
        row.pack(fill="x")
        self.escape_var = tk.BooleanVar(value=True)
        self.escape_check = ttk.Checkbutton(
            row, text="脱钩检测", variable=self.escape_var)
        self.escape_check.pack(side="left")
        self.calib_btn = self._flat_button(
            row, "框选提示区", lambda: self.calibrate("hud"))
        self.calib_btn.pack(side="left", padx=8)
        self.calib_var = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.calib_var,
                  style="Muted.TLabel").pack(side="left")
        ttk.Label(inner, style="Muted.TLabel",
                  text="线抛进水里（出现「刺鱼」提示）→ 点「框选提示区」→ 3 秒后框选提示文字"
                  ).pack(anchor="w", pady=(4, 0))

        row2 = ttk.Frame(inner)
        row2.pack(fill="x", pady=(6, 0))
        self.splash_var = tk.BooleanVar(value=False)
        self.splash_check = ttk.Checkbutton(
            row2, text="水花检测（看画面识别咬钩，不受旁人声音影响）",
            variable=self.splash_var)
        self.splash_check.pack(side="left")
        self.splash_btn = self._flat_button(
            row2, "框选水花区", lambda: self.calibrate("splash"))
        self.splash_btn.pack(side="left", padx=8)
        self.splash_calib_var = tk.StringVar(value="")
        ttk.Label(row2, textvariable=self.splash_calib_var,
                  style="Muted.TLabel").pack(side="left")
        ttk.Label(inner, style="Muted.TLabel",
                  text="正常钓着鱼 → 点「框选水花区」→ 3 秒后框住你自己浮漂的落点水域（别把别人的浮漂框进来）"
                  ).pack(anchor="w", pady=(4, 0))

        if not HAS_VISION:
            self.escape_check.config(state="disabled")
            self.calib_btn.config(state="disabled")
            self.splash_check.config(state="disabled")
            self.splash_btn.config(state="disabled")
            self.calib_var.set("缺库：pip install mss opencv-python")
        else:
            self._update_calib_status()

    def _update_calib_status(self):
        ok = HUD_TPL_PATH.exists() and HUD_REGION_PATH.exists()
        self.calib_var.set("已校准 ✓" if ok else "未校准")
        self.splash_calib_var.set(
            "已框选 ✓" if SPLASH_REGION_PATH.exists() else "未框选")

    def calibrate(self, mode="hud"):
        if self.bot and self.bot.is_alive():
            self._log("请先停止钓鱼再校准。")
            return
        tips = {
            "hud": "校准：3 秒后自动截取屏幕，请切到游戏并确保显示「刺鱼」提示……",
            "splash": "框选水花区：3 秒后自动截取屏幕，请切到游戏（正常钓鱼画面即可）……",
        }
        tip = tips.get(mode, tips["hud"])
        self._log(tip)
        self.withdraw()
        threading.Thread(target=self._calib_grab, args=(mode,), daemon=True).start()

    def _calib_grab(self, mode="hud"):
        time.sleep(3)
        try:
            with mss.mss() as sct:
                mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                img = np.asarray(sct.grab(dict(mon)))
        except Exception as exc:
            self.q.put(("log", f"截图失败：{exc}"))
            self.q.put(("calib_cancel",))
            return
        self.q.put(("calib_show", (img, mon, mode)))

    def _show_selector(self, img, mon, mode="hud"):
        """全屏显示截图，鼠标拖框选择区域。Esc 取消。"""
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        ok, png = cv2.imencode(".png", bgr)
        photo = tk.PhotoImage(data=base64.b64encode(png))

        tip_texts = {
            "hud": "按住左键拖框选中「刺鱼」提示文字，松开完成，Esc 取消",
            "splash": "按住左键框住你自己浮漂的落点水域（别把别人的浮漂框进来），松开完成，Esc 取消",
        }
        tip_text = tip_texts.get(mode, tip_texts["hud"])

        top = tk.Toplevel(self)
        top.overrideredirect(True)
        top.attributes("-topmost", True)
        top.geometry(f"{mon['width']}x{mon['height']}+{mon['left']}+{mon['top']}")
        canvas = tk.Canvas(top, width=mon["width"], height=mon["height"],
                           highlightthickness=0, cursor="crosshair")
        canvas.pack(fill="both", expand=True)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.photo = photo
        shadow = canvas.create_text(
            mon["width"] // 2 + 2, 42, anchor="center",
            text=tip_text,
            fill="#000000", font=("Microsoft YaHei UI", 15, "bold"))
        hint = canvas.create_text(
            mon["width"] // 2, 40, anchor="center",
            text=tip_text,
            fill="#ffffff", font=("Microsoft YaHei UI", 15, "bold"))
        canvas.tag_lower(shadow, hint)

        sel = {"x0": 0, "y0": 0, "rect": None}

        def cancel(_=None):
            top.destroy()
            self.deiconify()
            self._log("已取消校准。")

        def on_press(e):
            sel["x0"], sel["y0"] = e.x, e.y
            if sel["rect"]:
                canvas.delete(sel["rect"])
            sel["rect"] = canvas.create_rectangle(
                e.x, e.y, e.x, e.y, outline="#5ac890", width=2)

        def on_drag(e):
            canvas.coords(sel["rect"], sel["x0"], sel["y0"], e.x, e.y)

        def on_release(e):
            x0, x1 = sorted((sel["x0"], e.x))
            y0, y1 = sorted((sel["y0"], e.y))
            w, h = x1 - x0, y1 - y0
            if w < 40 or h < 25:
                self._log(f"框选区域太小（{w}x{h}），请按住左键拖出一个框（不是点一下）。")
                return
            region = {"left": int(mon["left"] + x0), "top": int(mon["top"] + y0),
                      "width": int(w), "height": int(h)}
            if mode == "hud":
                sub = img[y0:y1, x0:x1]
                gray = cv2.cvtColor(sub, cv2.COLOR_BGRA2GRAY)
                cv2.imwrite(str(HUD_TPL_PATH), gray)
                HUD_REGION_PATH.write_text(json.dumps(region), encoding="utf-8")
            elif mode == "splash":
                SPLASH_REGION_PATH.write_text(json.dumps(region), encoding="utf-8")
            top.destroy()
            self.deiconify()
            self._update_calib_status()
            self._log(f"框选完成 ✓ 区域 {w}x{h} @ ({region['left']}, {region['top']})")

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        top.bind("<Escape>", cancel)
        top.focus_set()

    # ---------- 高级参数（默认折叠） ----------

    def _build_param_section(self, parent):
        bar = ttk.Frame(parent)
        tk.Frame(parent, bg=C_LINE, height=1).pack(fill="x", pady=(12, 0))
        bar.pack(fill="x", pady=(6, 0))
        self.params_toggle = self._flat_button(
            bar, "高级参数 ▸", self._toggle_params)
        self.params_toggle.pack(side="left")
        ttk.Label(bar, text="默认值已调好，一般不用动",
                  style="Muted.TLabel").pack(side="left", padx=10)

        self.params_frame = ttk.Frame(parent)  # 默认不 pack = 折叠

        self.entries = {}
        specs = [
            ("threshold",          "咬钩阈值",        "0.58", "相似度超过它就收竿；误收就调高，听不到就调低"),
            ("ignore_seconds",     "抛竿后静音期(秒)", "5.0", "抛竿后多久内不监听，过滤抛竿噪音"),
            ("max_cast_seconds",   "最长等咬钩(秒)",  "20",  "超过就强制收竿"),
            ("second_press_delay", "第二次按键延迟(秒)", "1.5", "抛竿后隔多久再按一次"),
            ("react_min",          "反应延迟下限(秒)", "0.214", "咬钩后随机延迟收竿（防检测）"),
            ("react_max",          "反应延迟上限(秒)", "0.578", ""),
            ("anim_min",           "动画等待下限(秒)", "7.0",  "空竿收竿后等多久（跳过动画流程关闭时用）"),
            ("anim_max",           "动画等待上限(秒)", "9.0",  ""),
            ("skip_delay",         "跳过动画延迟(秒)", "2.5",  "钓到鱼收竿后隔几秒按键跳过展示动画"),
            ("post_skip_delay",    "跳过后等待(秒)",   "1.5",  "跳过动画后隔多久抛下一竿（太快游戏会拒收）"),
            ("bait_every",         "每几轮上饵",       "5",    "每钓这么多轮，停下来等上饵；填 0 关闭"),
            ("rebait_delay",       "上饵等待(秒)",     "5.0",  "到轮数后等多久让游戏上饵"),
            ("miss_skip_delay",    "空竿跳过延迟(秒)", "1.5",  "空竿收竿后隔几秒按键跳过收线动画"),
            ("escape_threshold",   "模板匹配阈值",     "0.55", "提示区和校准模板的相似度低于它就怀疑脱钩"),
            ("escape_grace",       "偏离确认时长(秒)", "1.2",  "提示区持续偏离这么久才判定脱钩"),
            ("splash_threshold",   "水花白色像素阈值", "120",  "水花区里亮白像素超过它就判定咬钩；暗场景浮漂假动作约70，真咬钩190+；误提就调高，不提就调低"),
            ("splash_mute",        "水花静音期(秒)",   "3.0",  "入水后头几秒不看水花，过滤甩杆入水的水花"),
            ("splash_jump",        "水花跳变触发量",   "50",   "白色像素单次增加超过它就判定咬钩（最快通道）"),
            ("splash_jump_min",    "跳变触发下限",     "60",   "跳变触发时白色像素至少要达到这个数，防假动作误提"),
            ("hold_seconds",       "按键按住时长(秒)", "0.05", ""),
            ("timeout_alert",      "连续超时弹窗",     "3",    "连续这么多轮没咬钩就弹窗+响铃提醒；0 关闭"),
            ("action_key",         "动作键",          "f6",   "游戏里抛竿/收竿绑定的键"),
            ("stop_key",           "停止热键",        "f8",   "随时按它停止"),
            ("countdown",          "启动倒计时(秒)",   "5",    "开始后留给你切回游戏的时间"),
        ]
        for i, (key, label, default, tip) in enumerate(specs):
            row, col = divmod(i, 4)
            cell = ttk.Frame(self.params_frame)
            cell.grid(row=row, column=col, sticky="w", padx=(0, 14), pady=3)
            lbl = ttk.Label(cell, text=label, style="Muted.TLabel")
            lbl.pack(anchor="w")
            if tip:
                self._attach_tip(lbl, tip)
            var = tk.StringVar(value=default)
            self._entry(cell, var, width=9).pack(anchor="w")
            self.entries[key] = var

        self.skip_anim_var = tk.BooleanVar(value=True)
        self.skip_anim_miss_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.params_frame, variable=self.skip_anim_var,
            text="钓到鱼后按键跳过展示动画"
        ).grid(row=(len(specs) + 3) // 4, column=0, columnspan=2,
               sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            self.params_frame, variable=self.skip_anim_miss_var,
            text="空竿收竿也按键跳过收线动画"
        ).grid(row=(len(specs) + 3) // 4, column=2, columnspan=2,
               sticky="w", pady=(8, 0))
        self.hold_rmb_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.params_frame, variable=self.hold_rmb_var,
            text="抛竿后按住右键缩放（浮漂聚焦屏幕中心；开了它要在缩放状态下框选水花区）"
        ).grid(row=(len(specs) + 3) // 4 + 1, column=0, columnspan=4,
               sticky="w", pady=(2, 0))
        self.sound_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.params_frame, variable=self.sound_var,
            text="声音检测（听咬钩音效）"
        ).grid(row=(len(specs) + 3) // 4 + 2, column=2, columnspan=2,
               sticky="w", pady=(2, 0))

    def _toggle_params(self):
        if self._params_visible:
            self.params_frame.pack_forget()
            self.params_toggle.config(text="高级参数 ▸")
            self._resize_window(-self._params_h)
        else:
            self.params_frame.pack(fill="x", pady=(2, 0))
            self.params_toggle.config(text="高级参数 ▾")
            self.update_idletasks()
            self._params_h = self.params_frame.winfo_reqheight() + 8
            self._resize_window(self._params_h)
        self._params_visible = not self._params_visible

    _params_h = 0

    def _resize_window(self, dh):
        """窗口高度增减 dh 像素，不超出屏幕。"""
        w = self.winfo_width()
        h = self.winfo_height() + dh
        h = max(540, min(h, self.winfo_screenheight() - 60))
        self.geometry(f"{w}x{h}")

    def _attach_tip(self, widget, text):
        tip = {"win": None}

        def show(_):
            if tip["win"]:
                return
            win = tk.Toplevel(widget)
            win.wm_overrideredirect(True)
            win.wm_geometry(f"+{widget.winfo_rootx()}+{widget.winfo_rooty() + 20}")
            tk.Label(win, text=text, background="#fdf6e3", foreground=C_TEXT,
                     relief="solid", borderwidth=1, font=F_SUB,
                     wraplength=260, justify="left",
                     padx=6, pady=4).pack()
            tip["win"] = win

        def hide(_):
            if tip["win"]:
                tip["win"].destroy()
                tip["win"] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    # ---------- 状态与日志 ----------

    def _build_status_section(self, parent):
        inner = self._section(parent, "状态")
        top = ttk.Frame(inner)
        top.pack(fill="x")
        self.state_var = tk.StringVar(value="就绪")
        ttk.Label(top, textvariable=self.state_var,
                  style="State.TLabel").pack(side="left")
        self.stats_var = tk.StringVar(value="轮数 0 · 咬钩 0 · 空竿 0")
        ttk.Label(top, textvariable=self.stats_var,
                  style="Muted.TLabel").pack(side="right", pady=(3, 0))
        bar_row = ttk.Frame(inner)
        bar_row.pack(fill="x", pady=(6, 0))
        self.score_var = tk.DoubleVar(value=0.0)
        self.score_text = tk.StringVar(value="0.00")
        self.score_bar = ttk.Progressbar(
            bar_row, variable=self.score_var, maximum=1.0,
            style="Accent.Horizontal.TProgressbar")
        self.score_bar.pack(side="left", fill="x", expand=True)
        ttk.Label(bar_row, textvariable=self.score_text,
                  style="Muted.TLabel", width=6).pack(side="left", padx=(8, 0))

    def _build_log_section(self, parent):
        tk.Frame(parent, bg=C_LINE, height=1).pack(fill="x", pady=(12, 0))
        self.log_text = tk.Text(
            parent, height=8, state="disabled", wrap="word",
            bg=C_LOG_BG, fg=C_LOG_FG, font=F_LOG,
            insertbackground=C_LOG_FG, relief="flat",
            padx=10, pady=8, spacing1=2)
        self.log_text.pack(fill="both", expand=True, pady=(8, 0))

    def _build_buttons(self):
        tk.Frame(self, bg=C_LINE, height=1).pack(fill="x", side="bottom")
        row = tk.Frame(self, bg=C_BG)
        row.pack(fill="x", side="bottom", padx=22, pady=12)
        self.start_btn = tk.Button(
            row, text="开始钓鱼", command=self.start,
            bg=C_DARK, fg="white", activebackground="#3a3835",
            activeforeground="white", font=F_BTN, relief="flat",
            cursor="hand2", pady=10, disabledforeground="#6b6862")
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 8))
        self.stop_btn = tk.Button(
            row, text="停止", command=self.stop, width=10,
            bg=C_FIELD, fg=C_TEXT, activebackground=C_LINE,
            font=F_BTN, relief="flat", cursor="hand2", pady=10,
            highlightthickness=1, highlightbackground=C_LINE,
            state="disabled", disabledforeground=C_MUTED)
        self.stop_btn.pack(side="left")

    # ---------- 配置持久化 ----------

    def _save_config(self):
        try:
            data = {
                "device": self.device_var.get(),
                "bite_template": self.bite_var.get(),
                "entries": {k: v.get() for k, v in self.entries.items()},
                "checks": {
                    "skip_anim": bool(self.skip_anim_var.get()),
                    "skip_anim_miss": bool(self.skip_anim_miss_var.get()),
                    "escape_detect": bool(self.escape_var.get()),
                    "hold_rmb": bool(self.hold_rmb_var.get()),
                    "splash_detect": bool(self.splash_var.get()),
                    "sound_detect": bool(self.sound_var.get()),
                },
            }
            CONFIG_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_config(self):
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        try:
            names = [s.name for s in self.speakers]
            if data.get("device") in names:
                self.device_var.set(data["device"])
            if data.get("bite_template"):
                self.bite_var.set(data["bite_template"])
            for k, v in data.get("entries", {}).items():
                if k in self.entries:
                    self.entries[k].set(str(v))
            checks = data.get("checks", {})
            if "skip_anim" in checks:
                self.skip_anim_var.set(checks["skip_anim"])
            if "skip_anim_miss" in checks:
                self.skip_anim_miss_var.set(checks["skip_anim_miss"])
            if "escape_detect" in checks:
                self.escape_var.set(checks["escape_detect"])
            if "hold_rmb" in checks:
                self.hold_rmb_var.set(checks["hold_rmb"])
            if "splash_detect" in checks:
                self.splash_var.set(checks["splash_detect"])
            if "sound_detect" in checks:
                self.sound_var.set(checks["sound_detect"])
        except Exception:
            pass

    # ---------- 控制 ----------

    def _read_config(self):
        cfg = {}
        floats = ["threshold", "ignore_seconds",
                  "max_cast_seconds", "second_press_delay", "react_min", "splash_mute", "splash_jump", "splash_jump_min",
                  "react_max", "anim_min", "anim_max", "hold_seconds",
                  "skip_delay", "post_skip_delay", "rebait_delay",
                  "miss_skip_delay", "escape_threshold", "escape_grace"]
        ints = ["countdown", "bait_every", "timeout_alert", "splash_threshold"]
        strs = ["action_key", "stop_key"]
        for key in floats:
            cfg[key] = float(self.entries[key].get())
        for key in ints:
            cfg[key] = int(self.entries[key].get())
        for key in strs:
            cfg[key] = self.entries[key].get().strip().lower()

        name = self.device_var.get()
        speaker = next((s for s in self.speakers if s.name == name), None)
        if speaker is None:
            raise RuntimeError("没有可用的声音设备（远程桌面请开声音重定向）")
        cfg["device_id"] = speaker.id
        cfg["device_name"] = speaker.name

        cfg["bite_template"] = self.bite_var.get().strip()
        cfg["skip_anim"] = bool(self.skip_anim_var.get())
        cfg["skip_anim_miss"] = bool(self.skip_anim_miss_var.get())
        cfg["sound_detect"] = bool(self.sound_var.get())
        cfg["escape_detect"] = bool(self.escape_var.get())
        cfg["hold_rmb"] = bool(self.hold_rmb_var.get())
        cfg["splash_detect"] = bool(self.splash_var.get())
        return cfg

    def start(self):
        try:
            cfg = self._read_config()
        except Exception as exc:
            self._log(f"参数错误：{exc}")
            return
        self._save_config()
        self.bot = FishingBot(cfg, self.q)
        self.bot.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

    def stop(self):
        if self.bot:
            self.bot.stop_event.set()

    def _on_close(self):
        self._save_config()
        self.stop()
        self.destroy()

    # ---------- 队列泵 ----------

    def _log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", time.strftime("%H:%M:%S ") + msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _alert(self, msg):
        """连续超时等情况的强提醒：响铃 + 弹窗 + 窗口提到最前。"""
        self._log("⚠ " + msg)
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            pass
        try:
            self.deiconify()
            self.lift()
            self.attributes("-topmost", True)
            self.attributes("-topmost", False)
        except Exception:
            pass
        messagebox.showwarning("自动钓鱼提醒", msg, parent=self)

    def _poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log(msg[1])
                elif kind == "score":
                    self.score_var.set(max(0.0, msg[1]))
                    self.score_text.set(f"{msg[1]:.2f}")
                elif kind == "stats":
                    c, got, miss = msg[1]
                    self.stats_var.set(f"轮数 {c} · 咬钩 {got} · 空竿 {miss}")
                elif kind == "state":
                    self.state_var.set(msg[1])
                elif kind == "calib_done":
                    self._update_calib_status()
                elif kind == "calib_show":
                    self._show_selector(*msg[1])
                elif kind == "alert":
                    self._alert(msg[1])
                elif kind == "calib_cancel":
                    self.deiconify()
                elif kind == "done":
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)


if __name__ == "__main__":
    App().mainloop()