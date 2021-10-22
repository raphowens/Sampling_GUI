from tkinter import *
from PIL import ImageTk, Image
from tkinter import ttk, filedialog, simpledialog, messagebox
from numpy import arange
import numpy as np
import pandas as pd
from math import ceil, floor
import random
#import xlsxwriter
import matplotlib.pyplot as plt
import seaborn as sns
"""import warnings
warnings.filterwarnings('ignore')"""

'''Python GUI TITLE AND GEOMETRY'''
root = Tk()
root.title('SAMPLING')
#root.geometry('1500x750')
width_value = root.winfo_screenwidth()
height_value = root.winfo_screenheight()
root.geometry('%dx%d+0+0' % (width_value, height_value))


'''GUI FRAME FOR THE FLASHLIST SECTION'''
frame = LabelFrame(root, text='Population/Samples Info')
frame.place(relx=0.01, rely=0.5, height=300, width=650)

'''SCROLLBAR FOR THE FLASHLIST FRAME'''
tv1 = ttk.Treeview(frame)
tv1.place(relheight=1, relwidth=1)
treescrolly = Scrollbar(frame, orient='vertical', command=tv1.yview)
treescrollx = Scrollbar(frame, orient='horizontal', command=tv1.xview)
tv1.configure(xscrollcommand=treescrollx, yscrollcommand=treescrolly)
treescrollx.pack(side='bottom', fill='x')
treescrolly.pack(side='right', fill='y')

'''FRAME FOR THE INFO SECTION'''
frame2 = LabelFrame(root, text='Statistics')
frame2.place(relx=0.56, rely=0.5, height=300, width=650)

'''SCROLLBAR FOR THE INFO FRAME'''
tv2 = ttk.Treeview(frame2)
tv2.place(relheight=1, relwidth=1)
treescrolly = Scrollbar(frame2, orient='vertical', command=tv2.yview)
treescrollx = Scrollbar(frame2, orient='horizontal', command=tv2.xview)
tv2.configure(xscrollcommand=treescrollx, yscrollcommand=treescrolly)
treescrollx.pack(side='bottom', fill='x')
treescrolly.pack(side='right', fill='y')

# SELECTION PROCESS FROM PALLETS AND MODULES
def selection1(sel_pal, sel_mod):
    global Sample1
    # RANDOM SELECTION OF PALLETS IN A CONTAINER
    Random_Pallet1 = Random_Lot1.groupby('Containers')['PalletSn'].apply(
        lambda x: np.random.choice(x.unique(), sel_pal, replace=False))

    dfs1 = []
    for lot_no in Random_Pallet1.index:
        for pallet_no in Random_Pallet1[lot_no]:
            dfs1.append(
                Population[
                    (Population['Containers'] == lot_no) & (Population['PalletSn'] == pallet_no)])

    Random_Lot_Pallet1 = pd.concat(dfs1)

    # RANDOM SELECTION OF MODULES IN A PALLET
    Sample1_ = Random_Lot_Pallet1.groupby(['Containers', 'PalletSn']).apply(
        lambda x: x.sample(sel_mod, replace=False))

    # RANDOM SELECTION OF SAMPLES FROM THE MODULES, SORTING AND EXPORTING TO CSV
    Sample1_ = Sample1_.sample(sample_no1)
    Sample1_.to_csv('Sample1.csv')
    Sample1 = pd.read_csv('Sample1.csv')
    Sample1 = Sample1.sort_values('Containers').reset_index()
    Sample1.index = arange(1, len(Sample1) + 1)
    Sample1 = Sample1[list(Population.columns)]

def selection2(sel_pal, sel_mod):
    global Sample2
    # RANDOM SELECTION OF PALLETS IN A CONTAINER
    Random_Pallet2 = Random_Lot2.groupby('Containers')['PalletSn'].apply(
        lambda x: np.random.choice(x.unique(), sel_pal, replace=False))

    dfs2 = []
    for lot_no in Random_Pallet2.index:
        for pallet_no in Random_Pallet2[lot_no]:
            dfs2.append(
                Population[
                    (Population['Containers'] == lot_no) & (Population['PalletSn'] == pallet_no)])

    Random_Lot_Pallet2 = pd.concat(dfs2)

    # RANDOM SELECTION OF MODULES IN A PALLET
    Sample2_ = Random_Lot_Pallet2.groupby(['Containers', 'PalletSn']).apply(
        lambda x: x.sample(sel_mod, replace=False))

    # RANDOM SELECTION OF SAMPLES FROM THE MODULES, SORTING AND EXPORTING TO CSV
    Sample2_ = Sample2_.sample(sample_no2)
    Sample2_.to_csv('Sample2.csv')
    Sample2 = pd.read_csv('Sample2.csv')
    Sample2 = Sample2.sort_values('Containers').reset_index()
    Sample2.index = arange(1, len(Sample2) + 1)
    Sample2 = Sample2[list(Population.columns)]



'''RANDOM SELECTION OF CONTAINERS'''
def sel_containers():
    global Random_Lot1, Random_Lot2, container_number

    # USER INPUT VALUE FOR NUMBER OF CONTAINERS
    container_number = simpledialog.askinteger('Input number', 'Enter number of containers')

    #  CONTAINERS SELECTION BASED ON MAX AND MIN POWER CLASS
    if container_number == 5 or container_number == 6:
        max_type = Population[(Population['Type']) == (Population['Type'].value_counts().idxmax())]
        min_type = Population[(Population['Type']) == (Population['Type'].value_counts().idxmin())]

        # UNIQUE CONTAINERS BASED ON POWER CLASS
        lot_max, lot_min = max_type['Containers'].unique(), min_type['Containers'].unique()

        # CONVERTING CONTAINERS TO LIST
        lot_max_list, lot_min_list = lot_max.tolist(), lot_min.tolist()

        # LENGTH OF CONTAINERS FOR MAX AND MIN POWER CLASS
        a, b = len(lot_max_list), len(lot_min_list)

        # CALCULATION TO GET 5 OR 6 CONTAINERS BASED ON RATIO
        x, y = (((a / (a + b)) * 100) / 10) / 2, (((b / (a + b)) * 100) / 10) / 2

        # CONDITIONS SO THAT A TOTAL OF 5 OR 6 CONTAINERS IS SELECTED BASED ON THE POWER CLASS
        if y == 0 or x == 0:
            y, x = int((ceil(((b / (a + b)) * 100) / 10)) / 2), int((ceil(((a / (a + b)) * 100) / 10)) / 2)

        elif y < 1:
            x, y = floor(x), ceil(y)

        elif x < 1:
            x, y = ceil(x), floor(y)

        elif x == y:
            x, y = int(ceil(x)), int(ceil(y))

        elif x > y:
            x, y = int(ceil(x)), int(floor(y))

        elif x < y:
            x, y = int(floor(x)), int(ceil(y))


        #RANDOM SELECTION OF THE REQUIRED NUMBER OF LOTS FROM ALL THE LOTS
        random_lot_max1, random_lot_min1 = random.sample(lot_max_list, x), random.sample(lot_min_list, y)

        random_lot_max2, random_lot_min2 = random.sample(lot_max_list, x), random.sample(lot_min_list, y)

        Random_Lot1 = random_lot_max1 + random_lot_min1

        Random_Lot1 = Population[Population['Containers'].isin(Random_Lot1)]

        Random_Lot2 = random_lot_max2 + random_lot_min2

        Random_Lot2 = Population[Population['Containers'].isin(Random_Lot2)]

    else:
        unique_values = Population['Containers'].unique()
        unique_lot_values = unique_values.tolist()

        Random_Lot1 = random.sample(unique_lot_values, container_number)
        Random_Lot2 = random.sample(unique_lot_values, container_number)

        Random_Lot1 = Population[Population['Containers'].isin(Random_Lot1)]
        Random_Lot2 = Population[Population['Containers'].isin(Random_Lot2)]


'''SAMPLE 1 SIZE SELECTION'''
def get_samples1():
    global Sample1, sample_no1
    sel_containers()

    # USER INPUT VALUE FOR SAMPLE SIZE
    sample_no1 = simpledialog.askinteger('Input number', 'Enter Sample Size 1')

    if 1 < sample_no1 <= 20:
        if container_number >= 5:
            selection1(2, 2)

        else:
            selection1(5, 4)

    elif 20 < sample_no1 <= 30:
        if container_number >= 5:
            selection1(3, 3)

        else:
            selection1(5, 6)


    elif 30 < sample_no1 <= 40:
        if container_number >= 5:

            selection1(4, 2)

        else:
            selection1(5, 8)


    elif 40 < sample_no1 <= 50:
        if container_number >= 5:

            selection1(5, 2)

        else:
            selection1(5, 10)


    elif 50 < sample_no1 <= 60:
        if container_number >= 5:

            selection1(4, 3)

        else:
            selection1(5, 12)


    elif 60 < sample_no1 <= 70:
        if container_number >= 5:

            selection1(3, 5)

        else:
            selection1(5, 14)


    elif 70 < sample_no1 <= 80:
        if container_number >= 5:
            selection1(4, 4)

        else:
            selection1(5, 16)


    elif 80 < sample_no1 <= 90:
        if container_number >= 5:
            selection1(3, 6)

        else:
            selection1(5, 16)


    elif 90 < sample_no1 <= 100:
        if container_number >= 5:

            selection1(4, 5)

        else:
            selection1(5, 18)


    elif sample_no1 > 100:
        if container_number >= 5:

            selection1(5, 6)
        else:
            selection1(5, 20)



    response = messagebox.showinfo('This is my popup', 'Sample 1 gotten')
    Label(root).pack()


'''SAMPLE 2 SIZE SELECTION'''
def get_samples2():
    global Sample2, sample_no2
    sel_containers()

    sample_no2 = simpledialog.askinteger('Input number', 'Enter Sample Size 2')
    if 0 < sample_no2 <= 20:
        if container_number >= 5:

            selection2(2, 2)

        else:
            selection2(5, 4)

    elif 20 < sample_no2 <= 30:
        if container_number >= 5:
            selection2(3, 2)

        else:
            selection2(5, 6)

    elif 30 < sample_no2 <= 40:
        if container_number >= 5:

            selection2(4, 2)

        else:
            selection2(5, 8)

    elif 40 < sample_no2 <= 50:
        if container_number >= 5:
            selection2(5, 2)

        else:
            selection2(5, 10)

    elif 50 < sample_no2 <= 60:
        if container_number >= 5:

            selection2(4, 3)

        else:
            selection2(5, 12)

    elif 60 < sample_no2 <= 70:
        if container_number >= 5:

            selection2(3, 5)

        else:
            selection2(5, 14)

    elif 70 < sample_no2 <= 80:
        if container_number >= 5:

            selection2(4, 4)

        else:
            selection2(5, 16)

    elif 80 < sample_no2 <= 90:
        if container_number >= 5:

            selection2(3, 6)

        else:
            selection2(5, 16)


    elif 90 < sample_no2 <= 100:
        if container_number >= 5:

            selection2(4, 5)

        else:
            selection2(5, 18)

    elif sample_no2 > 100:
        if container_number >= 5:

            selection2(5, 6)

        else:
            selection2(5, 20)

    response = messagebox.showinfo('This is my popup', 'Sample 2 gotten')
    Label(root).pack()


'''SCATTERED PLOT SHOWING SELECTED SAMPLED MODULES IN THE ENTIRE FLASHLIST '''
def plot_1(column1='PalletSn', column2='PM'):
    plt.figure(figsize=(30, 8))
    plt.title('Scattered Plot showing overall flashlist and samples')
    a = sns.scatterplot(x=column1, y=column2, data=Population, label='Population', s=20, color='grey')
    a = sns.scatterplot(x=column1, y=column2, data=Sample1, label='Sample1', marker='s', s=100)
    a = sns.scatterplot(x=column1, y=column2, data=Sample2, label='Sample2', marker='D', s=100)
    a.set(xticklabels=[])
    a.tick_params(bottom=False)
    plt.show()


'''KDE PLOT SHOWING THE DISTRIBUTION OF SELECTED SAMPLES AND THE FLASHLIST'''
def plot_2(column='PM'):
    plt.figure(figsize=(30, 8))
    plt.grid()
    plt.title('KDE plot of overall flashlist and samples')
    sns.kdeplot(data=Population, x=column, color='grey', label='Entire flashlist', linewidth=4, shade=True)
    sns.kdeplot(data=Sample1, x=column, color='r', label='Sample1', linewidth=4, shade=True)
    sns.kdeplot(data=Sample2, x=column, color='b', label='Sample2', linewidth=4, shade=True)
    plt.legend()
    plt.show()

'''SAVING/EXPORTING THE SELECTIED SAMPLES TO AN EXCEL FILE'''
def save_1():
    writer = pd.ExcelWriter('yyy_Sample.xlsx', engine='xlsxwriter')
    Sample1.to_excel(writer, sheet_name='Main_Sample')
    Sample2.to_excel(writer, sheet_name='Backup_Sample')
    writer.save()
    # writer.close()
    response = messagebox.showinfo('This is my popup', 'Sample 1 saved as Main Sample')
    Label(root).pack()


def save_2():
    writer = pd.ExcelWriter('yyy_Sample.xlsx', engine='xlsxwriter')
    Sample2.to_excel(writer, sheet_name='Main_Sample')
    Sample1.to_excel(writer, sheet_name='Backup_Sample')
    writer.save()
    # writer.close()
    response = messagebox.showinfo('This is my popup', 'Sample 2 saved as Main Sample')
    Label(root).pack()

'''FUNCTION OF THE STATISTICAL VALUES OF THE FLASHLIST AND SELECTED SAMPLES'''
def stats_population():
    clear_tree2()
    tv2.insert(parent='', index='end', text='Population Power statistics')
    tv2.insert(parent='', index='end', text=Population['PM'].describe())


def stats_sample1():
    clear_tree2()
    tv2.insert(parent='', index='end', text='First Sample Power statistics')
    tv2.insert(parent='', index='end', text=Sample1['PM'].describe())


def stats_sample2():
    clear_tree2()
    tv2.insert(parent='', index='end', text='Second Sample Power statistics')
    tv2.insert(parent='', index='end', text=Sample2['PM'].describe())

'''FUNCTION OF THE STATISTICAL VALUES OF THE FLASHLIST AND SELECTED SAMPLES'''
def corr():
    clear_tree2()

    d = Population['PM'].corr(Sample1['PM'])
    e = Population['PM'].corr(Sample2['PM'])
    tv2.insert(parent='', index='end', text=f' Power Correlation between Population and Sample 1 --- {d}')
    tv2.insert(parent='', index='end', text=f' Power Correlation between Population and Sample 2 --- {e}')

'''CLEAR FRAME OF PREVIOUS VALUE FOR A NEW VALUE'''
def clear_tree2():
    tv2.delete(*tv2.get_children())


def clear_tree1():
    tv1.delete(*tv1.get_children())

'''FLASHLIST AND SELECTED SAMPLES INFORMATION'''
def p_info():
    clear_tree1()
    tv1.insert('', 'end', 'A', text='The total number of containers')
    tv1.insert("A", "end", "A.1", text=Population['Containers'].nunique())

    tv1.insert("", "end", "B", text='The total number of pallets')
    tv1.insert("B", "end", "B.1", text=Population['PalletSn'].nunique())

    tv1.insert('', 'end', 'C', text='The total number of modules')
    tv1.insert("C", "end", "C.1", text=Population['SN'].nunique())

    tv1.insert('', 'end', 'D', text='The total number of Module Power Class')
    tv1.insert("D", "end", "D.1", text=Population.Type.value_counts().reset_index(name='Type Number'))


def s1_info():
    clear_tree1()
    tv1.insert('', 'end', 'A', text='The total number of containers in Sample1')
    tv1.insert("A", "end", "A.1", text=Sample1['Containers'].nunique())

    tv1.insert("", "end", "B", text='The total number of pallets in Sample1')
    tv1.insert("B", "end", "B.1", text=Sample1['PalletSn'].nunique())

    tv1.insert('', 'end', 'C', text='The total number of modules in Sample1')
    tv1.insert("C", "end", "C.1", text=Sample1['SN'].nunique())

    tv1.insert('', 'end', 'D', text='The total number of Module Power Class')
    tv1.insert("D", "end", "D.1", text=Sample1.Type.value_counts().reset_index(name='Type Number'))


def s2_info():
    clear_tree1()
    tv1.insert('', 'end', 'A', text='The total number of containers in Sample2')
    tv1.insert("A", "end", "A.1", text=Sample2['Containers'].nunique())

    tv1.insert("", "end", "B", text='The total number of pallets in Sample2')
    tv1.insert("B", "end", "B.1", text=Sample2['PalletSn'].nunique())

    tv1.insert('', 'end', 'C', text='The total number of modules in Sample2')
    tv1.insert("C", "end", "C.1", text=Sample2['SN'].nunique())

    tv1.insert('', 'end', 'D', text='The total number of Module Power Class')
    tv1.insert("D", "end", "D.1", text=Sample2.Type.value_counts().reset_index(name='Type Number'))

'''FILE OPEN FUNCTION TO SELECTED EXCEL FILE FOR SAMPLING'''
def file_open():
    global df
    global Population

    filename = filedialog.askopenfilename(
        title='Open A File',
        filetypes=[('Excel files', '*.xlsx')]
    )

    if filename:
        try:
            filename = r"{}".format(filename)
            df = pd.read_excel(filename)
            df = df.fillna(method='ffill')
            # Population_excel = pd.read_excel(filename)
            df = df[(list(df.columns))]


        except ValueError:
            my_label.config(text="File couldn't be opened.....try again")
        except FileNotFoundError:
            my_label.config(text="File couldn't be opened.....try again")

    # Clear old treeview
    clear_tree()

    # Set up new treeview
    my_tree['column'] = list(df.columns)
    my_tree['show'] = 'headings'

    # Loop through column list
    for column in my_tree['column']:
        my_tree.heading(column, text=column)

    # Put data in treeview
    df_rows = df.to_numpy().tolist()
    for row in df_rows:
        my_tree.insert("", "end", values=row)

    # Pack the treeview finally
    my_tree.pack()

    df.to_csv('Flashlist.csv')
    Population = pd.read_csv('Flashlist.csv')
    Population = Population[list(df.columns[1:])]


def clear_tree():
    my_tree.delete(*my_tree.get_children())


# Create TREEVIEW FRAME
my_frame = LabelFrame(root, text='Flashlist')
# my_frame.pack(pady=50, padx=50)
my_frame.place(height=250, width=1500, relx=0.01, rely=0.01)

# Create TREEVIEW SCROLLBAR
tree_scrolly = Scrollbar(my_frame, orient='vertical')
tree_scrolly.pack(side=RIGHT, fill=Y)

tree_scrollx = Scrollbar(my_frame, orient='horizontal')
tree_scrollx.pack(side=BOTTOM, fill=X)

# Create a TREEVIEW
my_tree = ttk.Treeview(my_frame, yscrollcommand=tree_scrolly.set, xscrollcommand=tree_scrollx.set)

# Configure the scrollbar
tree_scrolly.config(command=my_tree.yview)
tree_scrollx.config(command=my_tree.xview)

# Add a OptionMenu
my_menu = Menu(root)
root.config(menu=my_menu)

# Add menu dropdown
file_menu = Menu(my_menu, tearoff=False)
file_menu3 = Menu(my_menu, tearoff=0)

my_menu.add_cascade(label='File', menu=file_menu)
file_menu.add_command(label='Open', command=file_open)
file_menu.add_command(label='Population Info', command=p_info)
file_menu.add_command(label='Sample1 Info', command=s1_info)
file_menu.add_command(label='Sample2 Info', command=s2_info)

file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

random_menu = Menu(my_menu, tearoff=0)
my_menu.add_cascade(label='Randomize', menu=random_menu)

random_menu.add_command(label='Get Sample 1', command=get_samples1)
random_menu.add_command(label='Get Sample 2', command=get_samples2)

random_menu.add_command(label='Population Statistics', command=stats_population)
random_menu.add_command(label='Sample 1 Statistics', command=stats_sample1)
random_menu.add_command(label='Sample 2 Statistics', command=stats_sample2)
random_menu.add_command(label='Correlation', command=corr)

plot_menu = Menu(my_menu, tearoff=0)
my_menu.add_cascade(label='Plot', menu=plot_menu)
plot_menu.add_command(label='Scattered', command=plot_1)
plot_menu.add_command(label='Distribution', command=plot_2)

save_menu = Menu(my_menu, tearoff=0)
my_menu.add_cascade(label='Save', menu=save_menu)
save_menu.add_command(label='Sample 1', command=save_1)
save_menu.add_command(label='Sample 2', command=save_2)

my_label = Label(root, text='')
my_label.pack(pady=20)

root.mainloop()
