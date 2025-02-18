import tkinter as tk
from tkinter import ttk


class AIAssistantDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Assistant Dashboard")
        self.root.geometry("600x400")

        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.root, text="AI Assistant Activity Log", font=("Arial", 14)).pack(pady=10)

        self.log_text = tk.Text(self.root, height=15, width=70, state=tk.DISABLED)
        self.log_text.pack(padx=10, pady=5)

        self.refresh_button = ttk.Button(self.root, text="Refresh Log", command=self.refresh_log)
        self.refresh_button.pack(pady=10)

    def refresh_log(self):
        # Simulate log retrieval
        logs = [
            "[10:00] AI analyzed market trends",
            "[10:05] Strategy adjusted based on risk assessment",
            "[10:10] User requested financial report",
            "[10:15] AI suggested risk mitigation plan"
        ]

        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "\n".join(logs))
        self.log_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = AIAssistantDashboard(root)
    root.mainloop()
