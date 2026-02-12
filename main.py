"""
主入口文件 - 提供命令行界面
"""
import sys
import os


def print_banner():
    """打印横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🚗  智能汽车知识库问答系统                            ║
║                                                           ║
║     基于 LangChain + RAG + Streamlit                      ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """打印菜单"""
    print("\n请选择操作：\n")
    print("  1. 🚀 启动 Web 应用")
    print("  2. 🔧 初始化知识库")
    print("  3. 🧪 运行测试")
    print("  4. 📖 查看快速开始指南")
    print("  5. ❌ 退出")
    print()


def run_streamlit():
    """运行 Streamlit 应用"""
    print("\n正在启动 Web 应用...")
    os.system("streamlit run app.py")


def run_init():
    """运行初始化脚本"""
    print("\n正在初始化知识库...")
    os.system("python init_kb.py")


def run_test():
    """运行测试"""
    print("\n正在运行测试...")
    os.system("python test_rag.py")


def show_quickstart():
    """显示快速开始指南"""
    print("\n" + "="*60)
    print("📖 快速开始指南")
    print("="*60)
    print()
    print("步骤 1: 安装依赖")
    print("  uv pip install -r requirements.txt")
    print()
    print("步骤 2: 配置环境变量")
    print("  cp .env.example .env")
    print("  # 然后编辑 .env 文件，填写 OPENAI_API_KEY")
    print()
    print("步骤 3: 初始化知识库")
    print("  python init_kb.py")
    print()
    print("步骤 4: 启动应用")
    print("  streamlit run app.py")
    print()
    print("详细说明请查看: QUICKSTART.md")
    print("="*60)


def main():
    """主函数"""
    print_banner()
    
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command in ['run', 'start', 'app']:
            run_streamlit()
        elif command in ['init', 'initialize']:
            run_init()
        elif command in ['test']:
            run_test()
        elif command in ['help', '--help', '-h']:
            show_quickstart()
        else:
            print(f"未知命令: {command}")
            print("可用命令: run, init, test, help")
        return
    
    # 交互式菜单
    while True:
        print_menu()
        choice = input("请输入选项 (1-5): ").strip()
        
        if choice == '1':
            run_streamlit()
        elif choice == '2':
            run_init()
        elif choice == '3':
            run_test()
        elif choice == '4':
            show_quickstart()
        elif choice == '5':
            print("\n👋 再见！")
            break
        else:
            print("\n❌ 无效的选项，请重新输入")
        
        if choice in ['1', '2', '3']:
            input("\n按 Enter 键继续...")


if __name__ == "__main__":
    main()

