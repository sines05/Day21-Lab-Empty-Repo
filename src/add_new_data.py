import pandas as pd
import os

def main():
    phase1_path = "data/train_phase1.csv"
    phase2_path = "data/train_phase2.csv"
    
    if not os.path.exists(phase1_path) or not os.path.exists(phase2_path):
        print("Lỗi: Không tìm thấy file dữ liệu phase1 hoặc phase2.")
        return

    df1 = pd.read_csv(phase1_path)
    df2 = pd.read_csv(phase2_path)
    
    print(f"Số dòng Phase 1: {len(df1)}")
    print(f"Số dòng Phase 2: {len(df2)}")
    
    # Kết hợp dữ liệu
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    # Lưu đè lên phase1 để DVC tự động nhận diện thay đổi nội dung
    df_combined.to_csv(phase1_path, index=False)
    
    print(f"Đã hợp nhất dữ liệu. Tổng số dòng mới: {len(df_combined)}")
    print(f"Dữ liệu mới đã được lưu vào {phase1_path}")

if __name__ == "__main__":
    main()
