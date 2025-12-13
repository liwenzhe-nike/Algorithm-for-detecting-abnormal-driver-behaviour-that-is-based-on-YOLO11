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
        self.window.title("脑机接口实验室")  # 修改窗口标题
        self.window.geometry("300x200+0+0")  # 设置窗口大小和位置（左上角）
        self.window.attributes("-topmost", True)  # 窗口始终置顶
        self.window.attributes("-alpha", 0.8)  # 设置窗口透明度为80%

        # 添加带有“脑机接口实验室”字样的底色
        self.background_label = tk.Label(self.window, text="脑机接口实验室", font=("宋体", 12), bg="gray", fg="white")
        self.background_label.pack(fill=tk.BOTH, expand=True)

        # 添加动作提醒标签
        self.label = tk.Label(self.background_label, text="", font=("宋体", 18, "bold"), bg="gray", fg="white")
        self.label.pack(pady=20)

        # 添加自定义顺序输入框
        self.order_label = tk.Label(self.background_label, text="自定义顺序（如：1,3,5,2,4,6）：", font=("宋体", 10), bg="gray", fg="white")
        self.order_label.pack()

        self.order_entry = tk.Entry(self.background_label, width=20, bg="white", fg="black", font=("宋体", 10))
        self.order_entry.pack()

        # 创建开始按钮
        self.start_button = tk.Button(self.background_label, text="开始", command=self.start_reminder, bg="white", fg="black", borderwidth=0, highlightthickness=0, relief="flat", font=("宋体", 12))
        self.start_button.pack(pady=10)

        # 创建停止按钮
        self.stop_button = tk.Button(self.background_label, text="停止", command=self.stop_reminder, state=tk.DISABLED, bg="white", fg="black", borderwidth=0, highlightthickness=0, relief="flat", font=("宋体", 12))
        self.stop_button.pack(pady=10)

        self.running = False

    def start_app(self):
        """启动目标应用程序"""
        subprocess.Popen(self.app_path)

    def validate_custom_order(self, order_str):
        """验证自定义顺序是否有效"""
        try:
            order_list = [int(i) - 1 for i in order_str.split(",")]
            for idx in order_list:
                if idx < 0 or idx >= len(self.actions):
                    return False, "索引超出范围，请输入有效的数字序列（如：1,3,5,2,4,6）"
            return True, order_list
        except ValueError:
            return False, "输入无效，请输入数字序列（如：1,3,5,2,4,6）"

    def start_reminder(self):
        """启动提醒功能"""
        self.running = True
        self.start_app()  # 启动应用程序
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # 获取用户输入的自定义顺序
        custom_order_str = self.order_entry.get()
        if custom_order_str:
            is_valid, result = self.validate_custom_order(custom_order_str)
            if is_valid:
                self.custom_order = result
            else:
                self.label.config(text=result)
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

            if action_index < len(self.actions):
                action = self.actions[action_index]
                self.label.config(text=f"请做出动作：{action}")
            else:
                self.label.config(text="索引超出范围，请检查自定义顺序！")
                self.running = False
                break

            self.current_action_index = (self.current_action_index + 1) % len(self.actions)
            time.sleep(10)  # 每10秒提醒一次

    def run(self):
        """运行GUI"""
        self.window.mainloop()

# 示例：启动一个特定的应用程序（请替换为你的应用程序路径）
app_path = r"C:/Program Files/Tencent/WeChat/WeChat.exe"
reminder_app = ReminderApp(app_path)
reminder_app.run()