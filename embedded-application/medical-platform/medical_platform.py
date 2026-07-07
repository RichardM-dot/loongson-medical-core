#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙芯智慧医疗平台 —— GTK3 桌面应用（Vte 原生终端版）
基于龙架构 (LoongArch) + 本地大语言模型的智能病历撰写平台
"""

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Vte', '2.91')
gi.require_version('Pango', '1.0')
from gi.repository import Gtk, Gdk, GLib, Pango, Vte
import os

ASSISTANT_SRC  = "/home/richard/assistant.c"
ASSISTANT_BIN  = "/home/richard/assistant"
DESKTOP        = os.path.expanduser("~/Desktop")
LAUNCH_CMD     = "gcc -o " + ASSISTANT_BIN + " " + ASSISTANT_SRC + " && " + ASSISTANT_BIN

class MedicalPlatform(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="龙芯智慧医疗平台")
        self.set_default_size(1200, 700)
        self.connect("destroy", self.on_quit)

        # ── 主布局：左 | (中 | 右) ──
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

        # 开始按钮
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
        mid = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        mid.pack_start(Gtk.Label(label="<b>终端输出</b>", use_markup=True, xalign=0.0, margin_start=6), False, False, 0)

        self.vte = Vte.Terminal()
        self.vte.set_scrollback_lines(5000)
        self.vte.set_font(Pango.FontDescription.from_string("Monospace 10"))
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/home/richard/sense_voice_demo/lib"
        self.vte_env = [k + "=" + v for k, v in env.items()]
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

        # CSS
        css = b"#left-panel { background-color: #f0f3f7; }"
        sp = Gtk.CssProvider(); sp.load_from_data(css)
        Gtk.StyleContext.add_provider_for_screen(Gdk.Screen.get_default(), sp, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        self.show_all()

    # ─── 开始服务 ───
    def on_start(self, btn):
        self.btn_start.set_sensitive(False)
        self.btn_send.set_sensitive(True)
        self.btn_stop.set_sensitive(True)

        # 清空终端，启动编译 + 运行
        self.vte.reset(True, True)
        self.vte.spawn_sync(
            Vte.PtyFlags.DEFAULT,
            "/home/richard",
            ["/bin/bash", "-c", LAUNCH_CMD],
            self.vte_env,
            GLib.SpawnFlags.DO_NOT_REAP_CHILD,
            None, None, None
        )

    # ─── 发送回车 ───
    def on_send_enter(self, btn):
        self.vte.feed_child(b"\n")

    # ─── 停止服务 ───
    def on_stop(self, btn):
        # Ctrl+C 终止终端中运行的进程
        self.vte.feed_child(b"\x03")
        self._reset_buttons()

    def _reset_buttons(self):
        self.btn_start.set_sensitive(True)
        self.btn_send.set_sensitive(False)
        self.btn_stop.set_sensitive(False)

    # ─── Vte 右键复制 ───
    def _on_vte_click(self, widget, event):
        if event.button == 3:  # 右键
            menu = Gtk.Menu()
            copy_item = Gtk.MenuItem(label="复制")
            copy_item.connect("activate", lambda w: self.vte.copy_clipboard_format(1))  # 1 = TEXT
            menu.append(copy_item)
            menu.show_all()
            menu.popup_at_pointer(event)
            return True
        return False

    # ─── 右侧保存 ───
    def on_input_key(self, widget, event):
        if event.keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
            self.on_save(None)
            return True
        return False

    def on_save(self, btn):
        buf = self.tv_in.get_buffer()
        txt = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), True)
        if not txt.strip():
            return
        fname = os.path.join(DESKTOP, "病例.txt")
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(txt)
            buf.set_text("", -1)
        except Exception as e:
            pass

    def on_quit(self, win):
        Gtk.main_quit()

if __name__ == "__main__":
    MedicalPlatform()
    Gtk.main()
