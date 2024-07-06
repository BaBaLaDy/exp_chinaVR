import glob
import os
import pandas as pd

"""##########################################################################"""
ori_bool = True

ori_path = "Ori_motion"
Mod_path = "Modify_motion"
# if ori_bool:
#     paths = os.walk(ori_path)
# else:
#     paths = os.walk(Mod_path)

acc_fold_path_o = "./save_data/ori/acc_data/"
pos_fold_path_o = "./save_data/ori/pos_data/"
acc_fold_path_m = "./save_data/modify/acc_data/"
pos_fold_path_m = "./save_data/modify/pos_data/"

# if not os.path.exists(acc_fold_path_o):  # 判断所在目录下是否有该文件名的文件夹
#     os.makedirs(acc_fold_path_o)
# if not os.path.exists(pos_fold_path_o):
#     os.makedirs(pos_fold_path_o)
# if not os.path.exists(acc_fold_path_m):
#     os.makedirs(acc_fold_path_m)
# if not os.path.exists(pos_fold_path_m):
#     os.makedirs(pos_fold_path_m)
# merge_data = pd.DataFrame()
"""##########################################################################"""


# for file_name in file_names:
#     if file_name[-4:] == ".csv":
#         print(file_name)
#         file_path = os.path.join(fold_path, file_name)
#         data = pd.read_csv(file_path)
#         if file_name == "Waist.csv":
#             extra_data = data.iloc[:, 0:8]
#         else:
#             extra_data = data.iloc[:, 2:8]
#         merge_data = pd.concat([merge_data, extra_data], axis=1)
#
# merge_data.to_csv('merged_data.csv', index=False)

# def acc_process_func():


def merge_file_2_acc_pos(read_path, save_path, motion_name, type='_m'):
    """ merge files from unity and spilt them into acc and pos"""
    file_names = ["Waist.csv", "L_Upper_Leg.csv", "L_Lower_Leg.csv", "R_Upper_Leg.csv", "R_Lower_Leg.csv"]
    print(read_path)
    paths = glob.glob(read_path + '*' + motion_name + '*')
    acc_sum_data = pd.DataFrame()
    pos_sum_data = pd.DataFrame()
    print(paths)

    for num, dir_name in enumerate(paths):
        # if dir_lst:
        #     for dir_name in dir_lst:
        acc_merge_data = pd.DataFrame()
        pos_merge_data = pd.DataFrame()
        # file_path = os.path.join(read_path, dir_name)
        # print(file_path)
        # if len(file_names) != 0:
        for file_name in file_names:
            if file_name[-4:] == ".csv":
                print(file_name)
                file_path = os.path.join(dir_name, file_name)
                data = pd.read_csv(file_path, header=None, index_col=False)
                if file_name == "Waist.csv":
                    extra_data = data.iloc[:, :2]
                    acc_merge_data = pd.concat([acc_merge_data, extra_data], axis=1)
                    pos_merge_data = pd.concat([pos_merge_data, extra_data], axis=1)
                extra_data_acc = data.iloc[:, 2:5]
                extra_data_pos = data.iloc[:, 5:8]
                acc_merge_data = pd.concat([acc_merge_data, extra_data_acc], axis=1)
                pos_merge_data = pd.concat([pos_merge_data, extra_data_pos], axis=1)
        acc_merge_data = acc_merge_data.iloc[:-1]
        pos_merge_data = pos_merge_data.iloc[:-1]

        # acc_sum_data = pd.concat([acc_sum_data, acc_merge_data], axis=0)
        # pos_sum_data = pd.concat([pos_sum_data, pos_merge_data], axis=0)

        acc_file_name = motion_name + str(num) + '_acc_merge_data.csv'
        pos_file_name = motion_name + str(num) + '_pos_merge_data.csv'
        acc_save_path = os.path.join(save_path + 'acc/', acc_file_name)
        if not os.path.exists(save_path + 'acc/'):
            os.makedirs(save_path + 'acc/')
        pos_save_path = os.path.join(save_path + 'pos/', pos_file_name)
        if not os.path.exists(save_path + 'pos/'):
            os.makedirs(save_path + 'pos/')
        # acc_sum_data.to_csv(acc_save_path, index=False, header=False)
        # pos_sum_data.to_csv(pos_save_path, index=False, header=False)
        # acc_file_name = dir_name + '_m_acc_merge_data.csv'
        # pos_file_name = dir_name + '_m_pos_merge_data.csv'
        acc_merge_data.to_csv(acc_save_path, index=False, header=False)
        pos_merge_data.to_csv(pos_save_path, index=False, header=False)

    # if ori_bool:
    #     acc_file_name = dir_name + '_o_acc_merge_data.csv'
    #     pos_file_name = dir_name + '_o_pos_merge_data.csv'
    #     acc_save_path = os.path.join(acc_fold_path_o, acc_file_name)
    #     pos_save_path = os.path.join(pos_fold_path_o, pos_file_name)
    # else:

    # acc_save_path = os.path.join(acc_fold_path_m, acc_file_name)
    # pos_save_path = os.path.join(pos_fold_path_m, pos_file_name)

def modify_first_column(file_path):
    df = pd.read_csv(file_path, header=None)
    df.iloc[:, 0] = range(1, len(df) + 1)
    # df.to_csv(file_path, index=False)
    # df = pd.read_excel(file_path)
    df = df.iloc[:-1]
    df.to_csv(file_path, index=False, header=False)


# file_path = "save_data/ori/acc_data/highKnee_o_acc_merge_data.csv"
# modify_first_column(file_path)
if __name__ == "__main__":
    # data_path = "Animation_record/"
    # data_path = "./Virtual_New_New_xy_18/"
    # data_path = "./Virtual_New_addPelvis_xyz/"
    data_path = "./CorData-ori/scale_0123_trans/"

    # save_path = "./save_data/Vitural_NewData/"
    save_path = "./CorData/scale_0123_trans/"
    # save_path = "./save_data/Animation_record/"

    # data_path = "save_data/OriVirtual_Data/"
    # save_path = "./save_data/ori_Vitural_NewData/ankleTap/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dir_list = os.listdir(data_path)
    print(dir_list)
    for dir in dir_list:
        # motion_name = ['ankle', 'highKnee', 'Kneekick', 'reverseLunge', 'sideCrunch', 'sidetoside', 'warmup']
        motion_name = ['ReverseLunge', 'HighKnee', 'sidetoside']
        # motion_name = ['ankle', 'highKnee', 'reverseLunge', 'sideCrunch', 'sidetoside', 'warmup']
        # motion_name = ['reverseLunge']

        input_path = data_path + dir + '/'
        # input_path = data_path

        # output_path = save_path + dir + 'merge' + '/'
        output_path = save_path + dir + '/'

        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)
        for motion in motion_name:
            print(motion)
            merge_file_2_acc_pos(input_path, output_path, motion)

    # motion_name = ['ankle', 'highKnee', 'Kneekick', 'reverseLunge', 'sideCrunch', 'sidetoside', 'warmup']
    # for motion in motion_name:
    #     print(motion)
    #     save_path = "./save_data/Virtual_New_Conv/" + motion + "_conv/"
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     merge_file_2_acc_pos(data_path, save_path, motion)

