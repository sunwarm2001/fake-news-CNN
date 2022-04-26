from GUI_designer.UI import  *
from data_process import get_embedLookup
embed_lookup=''
def data_processing():
    #获取layout：GUI界面——数据处理
    window = sg.Window("基于CNN的虚假信息检测", layout=data_process())
    while True:
        event, values = window.read()
        #print(event, values)

        if event == sg.WIN_CLOSED:
            break
        elif event == '处理数据':
            true = values["true_data"]
            fake = values["fake_data"]
            word_model = values["word_model"]
            # print(type(true))
            # print(type(fake))
            # print(type(word_model))
            # print("true："+true)
            # print("fake："+fake)
            # print("word_model："+word_model)
            print("--------------开始数据处理---------------")
            embed_lookup = get_embedLookup(true, fake, word_model)
            print("--------------数据处理完成---------------")
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
            output_size = eval(['output_size'])
            num_filters = eval(['num_filters'])
            kernel_sizes = eval(['kernel_sizes'])
            dropout = eval(['dropout'])
            # print(epochs)
            # print(batch_size)
            # print(type(epochs))
            # print(type(batch_size))

            print("--------------开始训练模型---------------")

            print("--------------训练模型完成---------------")
    window.close()



if __name__ == '__main__':
    train_model()
