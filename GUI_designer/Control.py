from GUI_designer.UI import  *
from data_process import *
from model_CNN import *
embed_lookup=''
flag = 0

# GUI界面——数据处理
def data_processing():
    window = sg.Window("基于CNN的虚假信息检测", layout=data_process())
    while True:
        event, values = window.read()
        #print(event, values)

        if event == sg.WIN_CLOSED:
            break
        elif event == '处理数据':
            true_path = values["true_path"]
            fake_path = values["fake_path"]
            word_model_path = values["word_model_path"]
            # print(type(true))
            # print("true："+true)
            print("--------------开始数据处理---------------")
            get_embedLookup(true_path, fake_path, word_model_path)
            print("--------------数据处理完成---------------")
            flag = 1
        elif event == '分割数据' and flag == 0:
            sg.popup("请先执行数据处理")
        elif event == '分割数据' and flag == 1:
            train_size = eval(values["train_size"])
            test_size = eval(values["test_size"])
            random_state = eval(values['random_state'])
            # print(train_size)
            # print(type(train_size))
            print("--------------开始分割数据---------------")
            split_data(train_size, test_size, random_state)
            print("--------------分割数据完成---------------")
    window.close()

def train_model():
    window = sg.Window("基于CNN的虚假信息检测", layout=model_train())
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == '开始训练':
            # values[]得到的是字符串，要用eval将 str--> int
            epochs = eval(values['epochs'])
            batch_size = eval(values['batch_size'])
            lr = eval(values['lr'])
            output_size = eval(values['output_size'])
            num_filters = eval(values['num_filters'])
            kernel_sizes = values['kernel_sizes']
            kernel_sizes = [int(item) for item in kernel_sizes[1:-1].split(",")]
            dropout = eval(values['dropout'])
            # print(epochs)
            # print(batch_size)
            # print(kernel_sizes)
            # print(type(output_size))
            # print(type(num_filters))
            # print(type(kernel_sizes))
            net = CNN_model(output_size, num_filters, kernel_sizes, dropout)
            print("--------------开始训练模型---------------")
            train(net, epochs, lr, batch_size)
            print("--------------训练模型完成---------------")
    window.close()



if __name__ == '__main__':
    train_model()
