import pandas as pd


def view_jsonl_with_pandas(filename, n=100):
    """用pandas查看JSONL文件前n行"""
    try:
        # 读取前n行
        df = pd.read_json(filename, lines=True, nrows=n)

        print(f"文件: {filename}")
        print(f"形状: {df.shape} (行×列)")
        print(f"显示前 {len(df)} 行")
        print("=" * 80)

        # 显示前几行
        print(df.head())

        print("\n" + "=" * 80)
        print("📊 数据类型:")
        print(df.dtypes)

        print("\n📝 字段信息:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2}. {col:20} 非空值: {df[col].count()}/{len(df)}")
            if df[col].dtype == 'object':
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                print(f"    示例: {str(sample)[:50]}..." if sample else "    全为空")

    except Exception as e:
        print(f"读取失败: {e}")


# 使用
view_jsonl_with_pandas(r'D:\00_CityU-Data Science\SDSC6001\Project\elec_5core_meta.jsonl', 100000)