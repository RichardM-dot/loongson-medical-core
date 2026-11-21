#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙芯智慧医疗平台 —— GTK3 桌面应用（Vte 原生终端版）
基于龙架构 (LoongArch) + 本地大语言模型的智能病历撰写平台
"""

import gi
# 版本钉死：Vte 2.91 是 GTK3 配套的终端组件 API（2.90 之前接口不同）
gi.require_version('Gtk', '3.0')
gi.require_version('Vte', '2.91')
gi.require_version('Pango', '1.0')
from gi.repository import Gtk, Gdk, GLib, Pango, Vte
import os

# ── 固定路径与启动命令（部署在目标机的 /home/richard 下，换机需同步修改）──
ASSISTANT_SRC  = "/home/richard/assistant.c"   # 助手 C 源码（仓库 embedded-application/assistant.c 的部署副本）
ASSISTANT_BIN  = "/home/richard/assistant"     # 每次点"开始服务"都会重新编译出的可执行文件
DESKTOP        = os.path.expanduser("~/Desktop")  # 右侧面板"保存病历"的目标目录
# 一条命令完成"编译 + 运行"：源码改了无需手动 make，点按钮即生效
LAUNCH_CMD     = "gcc -o " + ASSISTANT_BIN + " " + ASSISTANT_SRC + " && " + ASSISTANT_BIN

class MedicalPlatform(Gtk.Window):
    # ── GUI 主窗口 ─────────────────────────────────────────────
    # 三面板布局：左 = 控制按钮 / 中 = Vte 原生终端(跑 assistant.c) / 右 = 病历粘贴保存。
    # 与 assistant.c 的耦合方式：往 Vte 终端喂回键(\n)或 Ctrl+C(\x03)，
    # 并依赖 assistant 打印的 [系统]/[你] 前缀文本在终端里展示状态。
    def __init__(self):
        Gtk.Window.__init__(self, title="龙芯智慧医疗平台")
        self.set_default_size(1200, 700)   # 宽 1200 保证三栏并排不挤
        self.connect("destroy", self.on_quit)

        # ── 主布局：左 | (中 | 右) ──
        # 嵌套 Paned：inner 包住"中|右"，outer 包住"左|inner"，
        # 这样三栏都可用鼠标拖动分隔条调宽。
        outer = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        inner = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self.add(outer)

        # ============ 左侧面板 ============
        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        left.set_margin_top(20); left.set_margin_bottom(20)
        left.set_margin_start(14); left.set_margin_end(14)
        left.set_size_request(270, -1)
        left.set_name("left-panel")

        lbl = Gtk.Label()
        lbl.set_markup("<b><big>龙芯智慧医疗平台</big></b>")
        lbl.set_halign(Gtk.Align.CENTER)
        left.pack_start(lbl, False, False, 0)

        desc = Gtk.Label()
        desc.set_markup(
            "本平台基于<span foreground='#1a73e8'><b>龙架构 (LoongArch)</b></span>优化，\n"
            "搭载<span foreground='#1a73e8'><b>本地大语言模型</b></span>，\n"
            "进行深度优化的<span foreground='#1a73e8'><b>智能病历撰写</b></span>。\n\n"
            "• 语音识别 → AI 推理\n"
            "• 全离线运行，隐私安全\n"
            "• 一键生成标准门诊病历\n\n"
            "<i>龙芯生态</i>"
        )
        desc.set_line_wrap(True); desc.set_xalign(0.0)
        left.pack_start(desc, True, True, 0)

        left.pack_start(Gtk.Separator(), False, False, 4)

        # 开始按钮：图标+文字组合，56px 高做成"主操作"视觉权重
        self.btn_start = Gtk.Button()
        b = Gtk.Box(spacing=6)
        b.pack_start(Gtk.Image.new_from_icon_name("media-playback-start", Gtk.IconSize.BUTTON), False, False, 0)
        b.pack_start(Gtk.Label(label="<big><b>开 始 服 务</b></big>", use_markup=True), True, True, 0)
        self.btn_start.add(b)
        self.btn_start.set_size_request(-1, 56)
        self.btn_start.connect("clicked", self.on_start)
        left.pack_start(self.btn_start, False, False, 0)

        # 发送回车
        self.btn_send = Gtk.Button(label="▸ 发送指令 (回车)")
        self.btn_send.set_sensitive(False)
        self.btn_send.connect("clicked", self.on_send_enter)
        left.pack_start(self.btn_send, False, False, 0)

        # 停止
        self.btn_stop = Gtk.Button(label="■ 停止服务")
        self.btn_stop.set_sensitive(False)
        self.btn_stop.connect("clicked", self.on_stop)
        left.pack_start(self.btn_stop, False, False, 0)

        outer.pack1(left, False, False)

        # ============ 中间面板（Vte 原生终端） ============
        # 用 Vte 而非 Gtk.TextView：Vte 是真终端（PTY），assistant.c 里的
        # getchar() 阻塞读回车、popen 子进程输出都能原生工作。
        mid = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        mid.pack_start(Gtk.Label(label="<b>终端输出</b>", use_markup=True, xalign=0.0, margin_start=6), False, False, 0)

        self.vte = Vte.Terminal()
        self.vte.set_scrollback_lines(5000)   # 多轮问诊的历史可回滚查看
        self.vte.set_font(Pango.FontDescription.from_string("Monospace 10"))
        env = os.environ.copy()
        # SenseVoice 的私有动态库路径：assistant.c 里 popen 拉起的
        # sense-voice-main 依赖它，必须在子进程环境里就位
        env["LD_LIBRARY_PATH"] = "/home/richard/sense_voice_demo/lib"
        self.vte_env = [k + "=" + v for k, v in env.items()]   # Vte 要求 KEY=VALUE 字符串列表
        mid.pack_start(self.vte, True, True, 0)

        # 右键菜单：复制
        self.vte.connect("button-press-event", self._on_vte_click)

        # ============ 右侧面板 ============
        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        right.set_margin_top(20); right.set_margin_bottom(20)
        right.set_margin_start(10); right.set_margin_end(14)
        right.set_size_request(290, -1)

        right.pack_start(Gtk.Label(label="<b>粘贴生成病历</b>", use_markup=True, halign=Gtk.Align.CENTER), False, False, 0)
        right.pack_start(Gtk.Label(label="<i>粘贴内容后按 Enter 保存到桌面</i>", use_markup=True, halign=Gtk.Align.CENTER), False, False, 0)

        sw2 = Gtk.ScrolledWindow()
        sw2.set_hexpand(True); sw2.set_vexpand(True)
        self.tv_in = Gtk.TextView()
        self.tv_in.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.tv_in.connect("key-press-event", self.on_input_key)
        sw2.add(self.tv_in)
        right.pack_start(sw2, True, True, 0)

        btn_save = Gtk.Button()
        sb = Gtk.Box(spacing=6)
        sb.pack_start(Gtk.Image.new_from_icon_name("document-save", Gtk.IconSize.BUTTON), False, False, 0)
        sb.pack_start(Gtk.Label(label="生成 病例.txt 到桌面"), True, True, 0)
        btn_save.add(sb)
        btn_save.set_size_request(-1, 50)
        btn_save.connect("clicked", self.on_save)
        right.pack_start(btn_save, False, False, 0)

        # 嵌套 Paned：中 | 右 放入 inner，然后 outer = 左 | inner
        inner.pack1(mid, True, False)
        inner.pack2(right, False, False)
        outer.pack2(inner, True, False)

        # CSS：只给左栏一点浅底色，区分"控制区"与"工作区"
        css = b"#left-panel { background-color: #f0f3f7; }"
        sp = Gtk.CssProvider(); sp.load_from_data(css)
        Gtk.StyleContext.add_provider_for_screen(Gdk.Screen.get_default(), sp, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        self.show_all()

    # ─── 开始服务 ───
    def on_start(self, btn):
        # 按钮状态机：开始置灰，发送/停止解锁（防重复 spawn）
        self.btn_start.set_sensitive(False)
        self.btn_send.set_sensitive(True)
        self.btn_stop.set_sensitive(True)

        # 清空终端，启动编译 + 运行
        self.vte.reset(True, True)   # True,True = 清屏 + 清回滚
        self.vte.spawn_sync(
            Vte.PtyFlags.DEFAULT,
            "/home/richard",                    # 子进程工作目录
            ["/bin/bash", "-c", LAUNCH_CMD],    # gcc 编译成功后立即运行 assistant
            self.vte_env,
            GLib.SpawnFlags.DO_NOT_REAP_CHILD,  # 交给 GLib 收尸，避免僵尸进程
            None, None, None
        )

    # ─── 发送回车 ───
    def on_send_enter(self, btn):
        # 往 PTY 里喂一个回车 = 告诉 assistant.c 的 getchar() "说完了"
        self.vte.feed_child(b"\n")

    # ─── 停止服务 ───
    def on_stop(self, btn):
        # Ctrl+C 终止终端中运行的进程
        self.vte.feed_child(b"\x03")   # ETX 字符，前台进程组收到 SIGINT
        self._reset_buttons()

    def _reset_buttons(self):
        # 恢复初始按钮态：开始可用，发送/停止置灰
        self.btn_start.set_sensitive(True)
        self.btn_send.set_sensitive(False)
        self.btn_stop.set_sensitive(False)

    # ─── Vte 右键复制 ───
    def _on_vte_click(self, widget, event):
        # 识别文本/病历内容需要复制时走这里（Vte 默认无右键菜单）
        if event.button == 3:  # 右键
            menu = Gtk.Menu()
            copy_item = Gtk.MenuItem(label="复制")
            copy_item.connect("activate", lambda w: self.vte.copy_clipboard_format(1))  # 1 = TEXT
            menu.append(copy_item)
            menu.show_all()
            menu.popup_at_pointer(event)
            return True   # 已处理，阻止事件继续传播
        return False

    # ─── 右侧保存 ───
    def on_input_key(self, widget, event):
        # 文本框里直接按 Enter（含小键盘 Enter）= 快捷保存
        if event.keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
            self.on_save(None)
            return True   # 吞掉按键事件，阻止文本框换行
        return False

    def on_save(self, btn):
        # 把右侧粘贴的病历文本整体写入桌面"病例.txt"（覆盖写）
        buf = self.tv_in.get_buffer()
        txt = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), True)   # True = 含不可见字符
        if not txt.strip():
            return   # 空内容不落盘
        fname = os.path.join(DESKTOP, "病例.txt")
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(txt)
            buf.set_text("", -1)   # 成功后清空输入框，提示"已保存"
        except Exception as e:
            pass   # 已知隐患：异常被静默吞掉，用户得不到保存失败提示

    def on_quit(self, win):
        Gtk.main_quit()   # 关窗即退出主循环；assistant 由自身 pidof 清场机制兜底

if __name__ == "__main__":
    MedicalPlatform()
    Gtk.main()
