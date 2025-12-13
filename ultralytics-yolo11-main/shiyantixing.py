import subprocess
import tkinter as tk
import time
import threading

class ReminderApp:
    def __init__(self, app_path):
        self.app_path = app_path  # 目标应用程序的路径
        self.actions = ["疲劳", "饮水", "玩手机", "犯困", "抽烟", "打哈欠"]
        self.current_action_index = 0
        self.custom_order = None  # 自定义顺序

        # 创建GUI窗口
        self.window = tk.Tk()
        self.window.title("动作提醒器")
        self.window.geometry("300x250")

        self.label = tk.Label(self.window, text="", font=("Arial", 14))
        self.label.pack(pady=20)

        self.order_label = tk.Label(self.window, text="自定义顺序（如：1,3,5,2,4,6）：", font=("Arial", 10))
        self.order_label.pack()

        self.order_entry = tk.Entry(self.window, width=20)
        self.order_entry.pack()

        self.start_button = tk.Button(self.window, text="开始", command=self.start_reminder)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.window, text="停止", command=self.stop_reminder, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.running = False

    def start_app(self):
        """启动目标应用程序"""
        subprocess.Popen(self.app_path)

    def start_reminder(self):
        """启动提醒功能"""
        self.running = True
        self.start_app()  # 启动应用程序
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # 获取用户输入的自定义顺序
        custom_order_str = self.order_entry.get()
        if custom_order_str:
            try:
                self.custom_order = [int(i) - 1 for i in custom_order_str.split(",")]
            except ValueError:
                self.label.config(text="自定义顺序无效，请输入数字序列（如：1,3,5,2,4,6）")
                self.running = False
                return

        threading.Thread(target=self.reminder_loop).start()

    def stop_reminder(self):
        """停止提醒功能"""
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.label.config(text="提醒已停止")

    def reminder_loop(self):
        """提醒循环"""
        while self.running:
            if self.custom_order:
                action_index = self.custom_order[self.current_action_index % len(self.custom_order)]
            else:
                action_index = self.current_action_index

            action = self.actions[action_index]
            self.label.config(text=f"请做出动作：{action}")
            self.current_action_index = (self.current_action_index + 1) % len(self.actions)
            time.sleep(10)  # 每10秒提醒一次

    def run(self):
        """运行GUI"""
        self.window.mainloop()

# 示例：启动一个特定的应用程序（请替换为你的应用程序路径）
app_path = r"C:/Program Files/Tencent/WeChat/WeChat.exe"
reminder_app = ReminderApp(app_path)
reminder_app.run()