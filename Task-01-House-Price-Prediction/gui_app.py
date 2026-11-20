"""
Modern Graphical User Interface (GUI) for House Price Prediction
Built with Tkinter (Native Python GUI)
"""

import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_models():
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
    multi_path = os.path.join(BASE_DIR, 'models', 'multiple_linear_regression.pkl')
    simple_path = os.path.join(BASE_DIR, 'models', 'simple_linear_regression.pkl')

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(multi_path, 'rb') as f:
        multi_model = pickle.load(f)
    with open(simple_path, 'rb') as f:
        simple_model = pickle.load(f)

    return scaler, multi_model, simple_model

class HousePricePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏠 House Price Predictor - Data Science Internship")
        self.root.geometry("640x740")
        self.root.resizable(False, False)
        self.root.configure(bg="#F4F6F9")

        # Load models
        try:
            self.scaler, self.multi_model, self.simple_model = load_models()
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load trained models:\n{e}")
            self.root.destroy()
            return

        self.create_widgets()

    def create_widgets(self):
        # Header Frame
        header = tk.Frame(self.root, bg="#1E3A8A", pady=18)
        header.pack(fill="x")
        
        lbl_title = tk.Label(header, text="🏠 House Price Prediction System", font=("Segoe UI", 18, "bold"), fg="white", bg="#1E3A8A")
        lbl_title.pack()
        lbl_sub = tk.Label(header, text="Simple & Multiple Linear Regression ML Models", font=("Segoe UI", 10), fg="#93C5FD", bg="#1E3A8A")
        lbl_sub.pack(pady=2)

        # Main Form Frame
        form = tk.Frame(self.root, bg="#FFFFFF", padx=25, pady=20, relief="groove", bd=1)
        form.pack(padx=25, pady=15, fill="both", expand=True)

        self.entries = {}
        fields = [
            ("Living Area (Square Footage):", "sqft", "2200", "sq ft (e.g. 800 - 4500)"),
            ("Total Rooms Above Ground:", "rooms", "7", "e.g. 4 - 12 rooms"),
            ("Bedrooms:", "beds", "3", "e.g. 1 - 6 bedrooms"),
            ("Bathrooms:", "baths", "2.5", "e.g. 1.0, 2.0, 2.5, 3.5"),
            ("Year Built (Construction Year):", "year", "2015", "e.g. 1960 - 2024"),
            ("Lot Area (Square Footage):", "lot", "8500", "e.g. 3000 - 25000 sq ft"),
            ("Garage Capacity (Cars):", "garage", "2", "0, 1, 2, or 3 cars"),
            ("Overall Quality Rating (1 - 10):", "quality", "8", "Scale: 1 (Poor) to 10 (Luxury)")
        ]

        for i, (label_text, key, default_val, hint) in enumerate(fields):
            lbl = tk.Label(form, text=label_text, font=("Segoe UI", 10, "bold"), bg="#FFFFFF", fg="#1F2937", anchor="w")
            lbl.grid(row=i, column=0, sticky="w", pady=4)

            entry = ttk.Entry(form, font=("Segoe UI", 10), width=18)
            entry.insert(0, default_val)
            entry.grid(row=i, column=1, padx=10, pady=4, sticky="w")
            self.entries[key] = entry

            lbl_hint = tk.Label(form, text=hint, font=("Segoe UI", 8, "italic"), bg="#FFFFFF", fg="#6B7280", anchor="w")
            lbl_hint.grid(row=i, column=2, sticky="w", pady=4)

        # Button Frame
        btn_frame = tk.Frame(form, bg="#FFFFFF")
        btn_frame.grid(row=len(fields), column=0, columnspan=3, pady=16)

        btn_predict = tk.Button(
            btn_frame, 
            text="✨ Predict House Price ✨", 
            font=("Segoe UI", 12, "bold"), 
            bg="#2563EB", 
            fg="white", 
            activebackground="#1D4ED8", 
            activeforeground="white",
            padx=20, 
            pady=8, 
            relief="flat", 
            cursor="hand2", 
            command=self.predict_price
        )
        btn_predict.pack(side="left", padx=5)

        btn_reset = tk.Button(
            btn_frame,
            text="Reset to Defaults",
            font=("Segoe UI", 10),
            bg="#E5E7EB",
            fg="#374151",
            activebackground="#D1D5DB",
            padx=12,
            pady=8,
            relief="flat",
            cursor="hand2",
            command=self.reset_defaults
        )
        btn_reset.pack(side="left", padx=5)

        # Results Frame
        res_frame = tk.LabelFrame(self.root, text=" 📊 Estimated House Prices ", font=("Segoe UI", 11, "bold"), bg="#F8FAFC", fg="#0F172A", padx=15, pady=12)
        res_frame.pack(padx=25, pady=(0, 20), fill="x")

        # Result Labels
        self.lbl_simple = tk.Label(res_frame, text="Simple Linear Regression (SqFt only): $ ---", font=("Segoe UI", 11), bg="#F8FAFC", fg="#1E293B", anchor="w")
        self.lbl_simple.pack(fill="x", pady=2)

        self.lbl_multi = tk.Label(res_frame, text="Multiple Linear Regression (All Features): $ ---", font=("Segoe UI", 13, "bold"), bg="#F8FAFC", fg="#15803D", anchor="w")
        self.lbl_multi.pack(fill="x", pady=4)

        self.lbl_note = tk.Label(res_frame, text="Click 'Predict House Price' to compute the estimated market valuation.", font=("Segoe UI", 9, "italic"), bg="#F8FAFC", fg="#64748B", anchor="w")
        self.lbl_note.pack(fill="x")

    def reset_defaults(self):
        defaults = {"sqft": "2200", "rooms": "7", "beds": "3", "baths": "2.5", "year": "2015", "lot": "8500", "garage": "2", "quality": "8"}
        for k, v in defaults.items():
            self.entries[k].delete(0, tk.END)
            self.entries[k].insert(0, v)
        self.lbl_simple.config(text="Simple Linear Regression (SqFt only): $ ---")
        self.lbl_multi.config(text="Multiple Linear Regression (All Features): $ ---")
        self.lbl_note.config(text="Defaults restored. Click 'Predict House Price' to evaluate.")

    def predict_price(self):
        try:
            sqft = float(self.entries["sqft"].get())
            rooms = int(float(self.entries["rooms"].get()))
            beds = int(float(self.entries["beds"].get()))
            baths = float(self.entries["baths"].get())
            year = int(float(self.entries["year"].get()))
            lot = float(self.entries["lot"].get())
            garage = int(float(self.entries["garage"].get()))
            quality = int(float(self.entries["quality"].get()))

            # Validation
            if sqft <= 0 or rooms <= 0 or beds <= 0:
                raise ValueError("Square footage and room counts must be positive numbers.")
            if not (1 <= quality <= 10):
                raise ValueError("Quality rating must be between 1 and 10.")

            # Simple Linear Regression (SqFt only)
            simple_val = self.simple_model.predict(pd.DataFrame({'SquareFootage': [sqft]}))[0]

            # Multiple Linear Regression (All Features normalized)
            row = pd.DataFrame([{
                'SquareFootage': sqft,
                'TotalRooms': rooms,
                'Bedrooms': beds,
                'Bathrooms': baths,
                'YearBuilt': year,
                'LotArea': lot,
                'GarageCars': garage,
                'OverallQuality': quality
            }])
            scaled_row = self.scaler.transform(row)
            multi_val = self.multi_model.predict(scaled_row)[0]

            # Update GUI Display
            self.lbl_simple.config(text=f"Simple Linear Regression (SqFt only):   ${simple_val:,.2f}")
            self.lbl_multi.config(text=f"Multiple Linear Regression (All Features):   ${multi_val:,.2f}")
            self.lbl_note.config(text=f"Success! Model evaluated for a {sqft:,.0f} sq ft house built in {year} (Quality: {quality}/10).")

        except Exception as e:
            messagebox.showerror("Input Error", f"Please check your input values:\n{e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = HousePricePredictorApp(root)
    root.mainloop()
