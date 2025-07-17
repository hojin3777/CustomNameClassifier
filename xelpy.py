import xlwings

def main():
    wb = xlwings.Book.caller()
    sheet = wb.sheets[0]
    sheet["A1"].value = "Hello, Excel from Python!"

if __name__ == "__main__":
    path = "C:\code"
    xlwings.Book(path+"/"+"xelpy.xlsm").set_mock_caller()
    main()