import random
import numpy as np
from TableGeneration.Distribution import Distribution
import time

'''
The code for generating 4 categories of tables consists of several small pieces e.g. types of borders, 
irregular/regular headers and transformations.

We define header_categories which has two possiblities: 
1. Regular headers
2. Irregular headers

Both of header categories are equally likely to be chosen randomly.

Regular Headers: Tables with only first row containing headers
Irregular Headers: Tables which have headings on first column of each row.

We define border_categories with 4 possibilities: 
1. All borders 
2. No borders
3. Borders only under headings
4. Only internal borders

Border category 1 is fixed for tables of category 1 and there is no other option while for the rest of categories,
the table border can be of any 4 categories. Therefore, we randomly select a border category with equal probability
for all 4.

'''

class Table:

    def __init__(self,no_of_rows,no_of_cols,images_path,ocr_path,gt_table_path,assigned_category,distributionfile):

        #get distribution of data
        self.distribution=Distribution(images_path,ocr_path,gt_table_path,distributionfile)

        self.all_words,self.all_numbers,self.all_others=self.distribution.get_distribution()

        self.assigned_category=assigned_category

        self.no_of_rows=no_of_rows
        self.no_of_cols=no_of_cols
        self.header_categories = {'types': [0, 1], 'probs': [0.5, 0.5]}
        self.header_cat = random.choices(self.header_categories['types'], weights=self.header_categories['probs'])[0]

        if(self.assigned_category==1):
            self.border_cat=0
        elif(self.assigned_category==2):
            self.border_cat=1
        elif(self.assigned_category==3):
            self.border_cat=2
        else:
            self.borders_categories = {'types': [0,1, 2, 3], 'probs': [0.25,0.25, 0.25, 0.25]}
            self.border_cat = random.choices(self.borders_categories['types'], weights=self.borders_categories['probs'])[0]
        
        self.spanflag=False

        self.idcounter=0

        '''cell_types matrix have two possible values: 'n' and 'w' where 'w' means word and 'n' means number'''
        self.cell_types=np.chararray(shape=(self.no_of_rows,self.no_of_cols))

        '''headers matrix have two possible values: 's' and 'h' where 'h' means header and 's' means simple text'''
        self.headers=np.chararray(shape=(self.no_of_rows,self.no_of_cols))

        '''A positive value at a position in matrix shows the number of columns to span and -1 will show to skip that cell as part of spanned cols'''
        self.col_spans_matrix=np.zeros(shape=(self.no_of_rows,self.no_of_cols))

        '''A positive value at a position means number of rows to span and -1 will show to skip that cell as part of spanned rows'''
        self.row_spans_matrix=np.zeros(shape=(self.no_of_rows,self.no_of_cols))

        '''missing_cells will contain a list of (row,column) pairs where each pair would show a cell where no text should be written'''
        self.missing_cells=[]

        #header_count will keep track of how many top rows and how many left columns are being considered as headers
        self.header_count={'r':2,'c':0}

        '''This matrix is essential for generating same cell, same row and same col matrices. Because this 
        matrix holds the list of word ids in each cell of the table'''
        self.data_matrix = np.empty(shape=(self.no_of_rows,self.no_of_cols),dtype=object)

        

    def get_log_value(self):
        ''' returns log base 2 (x)'''
        import math
        return int(math.log(self.no_of_rows*self.no_of_cols,2))


    def define_col_types(self):
        '''
        We define the data type that will go in each column. We categorize data in three types:
        1. 'n': Numbers
        2. 'w': word
        3. 'r': other types (containing special characters)

        '''

        len_all_words=len(self.all_words)
        len_all_numbers=len(self.all_numbers)
        len_all_others=len(self.all_others)

        total = len_all_words+len_all_numbers+len_all_others

        prob_words = len_all_words / total
        prob_numbers = len_all_numbers / total
        prob_others=len_all_others/total

        for i,type in enumerate(random.choices(['n','w','r'], weights=[prob_numbers,prob_words,prob_others], k=self.no_of_cols)):
            self.cell_types[:,i]=type

        '''The headers should be of type word'''
        self.cell_types[0:2,:]='w'

        '''All cells should have simple text but the headers'''
        self.headers[:] = 's'
        self.headers[0:2, :] = 'h'


    def generate_random_text(self,type):
        '''Depending on the data type of column, this function returns a randomly selected string (words or numbers)
        from unlv dataset and unique id assigned to Each word or number in the string.
        '''
        html=''
        ids=[]
        if(type=='n'):
            out= random.sample(self.all_numbers,1)
        elif(type=='r'):
            out=random.sample(self.all_others,1)
        else:
            text_len=random.randint(1,2)
            out= random.sample(self.all_words,text_len)

        for e in out:
            html+='<span id='+str(self.idcounter)+'>'+str(e)+' </span>'
            ids.append(self.idcounter)
            self.idcounter+=1
        return html,ids


    def agnostic_span_indices(self,maxvalue,max_lengths=-1):
        '''Spans indices. Can be used for row or col span
        Span indices store the starting indices of row or col spans while span_lengths will store
        the length of span (in terms of cells) starting from start index.'''
        span_indices = []
        span_lengths = []
        span_count = random.randint(1, 3)
        if(span_count>=maxvalue):
            return [],[]

        indices = sorted(random.sample(list(range(0, maxvalue)), span_count))

        starting_index = 0
        for i, index in enumerate(indices):
            if (starting_index > index):
                continue

            max_lengths=maxvalue-index
            if(max_lengths<2):
                break
            len_span = random.randint(1, max_lengths)

            if (len_span > 1):
                span_lengths.append(len_span)
                span_indices.append(index)
                starting_index = index + len_span

        return span_indices, span_lengths


    def make_header_col_spans(self):
        '''This function spans header cells'''
        while(True):                                        #iterate until we get some row or col span indices
            header_span_indices, header_span_lengths = self.agnostic_span_indices(self.no_of_cols)
            if(len(header_span_indices)!=0 and len(header_span_lengths)!=0):
                break

        row_span_indices=[]
        for index,length in zip(header_span_indices,header_span_lengths):
            self.spanflag=True
            self.col_spans_matrix[0,index]=length
            self.col_spans_matrix[0,index+1:index+length]=-1
            row_span_indices+=list(range(index,index+length))

        b=list(filter(lambda x: x not in row_span_indices, list(range(self.no_of_cols))))
        self.row_spans_matrix[0,b]=2
        self.row_spans_matrix[1,b]=-1

        #If the table has irregular headers, then we can span some of the rows in those header cells
        if(self.header_cat==1):
            self.create_irregular_header()


    def create_irregular_header(self):
        '''To make some random row spans for headers on first col of each row'''

        colnumber=0
        #-2 to exclude top 2 rows of header and -1 so it won't occupy the complete column
        span_indices, span_lengths = self.agnostic_span_indices(self.no_of_rows-2)
        span_indices=[x+2 for x in span_indices]

        for index, length in zip(span_indices, span_lengths):
            self.spanflag=True
            self.row_spans_matrix[index,colnumber]=length
            self.row_spans_matrix[index+1:index+length,colnumber]=-1
        self.headers[:,colnumber]='h'
        self.header_count['c']+=1



    def generate_missing_cells(self):
        '''This is randomly select some cells to be empty (not containing any text)'''
        missing=np.random.random(size=(self.get_log_value(),2))
        missing[:,0]=(self.no_of_rows - 1 - self.header_count['r'])*missing[:,0]+self.header_count['r']
        missing[:, 1] = (self.no_of_rows -1 - self.header_count['c']) * missing[:, 1] + self.header_count['c']
        for arr in missing:
            self.missing_cells.append((int(arr[0]), int(arr[1])))


    def create_style(self):
        '''This function will dynamically create stylesheet. This stylesheet essentially creates our specific
        border types in tables'''

        style = "<head><style>"
        style += "html{width:1366px;height:768px;background-color: white;}table{"

        # random center align
        if (random.randint(0, 1) == 1):
            style += "text-align:center;"

        style += """border-collapse:collapse;}td,th{padding:6px;padding-left: 15px;padding-right: 15px;"""

        if(self.border_cat==0):
            style += """ border:1px solid black;} """
        elif(self.border_cat==2):
            style += """border-bottom:1px solid black;}"""
        elif(self.border_cat==3):
            style+="""border-left: 1px solid black;}
                       th{border-bottom: 1px solid black;} table tr td:first-child, 
                       table tr th:first-child {border-left: 0;}"""
        else:
            style+="""}"""

        style += "</style></head>"
        return style

    def create_html(self):
        '''Depending on various conditions e.g. columns spanned, rows spanned, data types of columns,
        regular or irregular headers, tables types and border types, this function creates equivalent html
        script'''

        temparr=['td', 'th']
        html="""<html>"""
        html+=self.create_style()
        html+="""<body><table>"""
        for r in range(self.no_of_rows):
            html+='<tr>'
            for c in range(self.no_of_cols):

                row_span_value = int(self.row_spans_matrix[r, c])
                col_span_value = int(self.col_spans_matrix[r, c])
                htmlcol = temparr[['s', 'h'].index(self.headers[r][c].decode('utf-8'))]

                if (row_span_value == -1):
                    self.data_matrix[r, c] = self.data_matrix[r - 1, c]
                    continue
                elif(row_span_value>0):
                    html += '<' + htmlcol + ' rowspan=\"' + str(row_span_value) + '"'
                else:
                    if(col_span_value==0):
                        if (r, c) in self.missing_cells:
                            html += """<td></td>"""
                            continue
                    if (col_span_value == -1):
                        self.data_matrix[r, c] = self.data_matrix[r, c - 1]
                        continue
                    html += '<' + htmlcol + """ colspan=""" + str(col_span_value)

                out,ids = self.generate_random_text(self.cell_types[r, c].decode('utf-8'))
                html+='>'+out+'</'+htmlcol+'>'

                self.data_matrix[r,c]=ids

            html += '</tr>'

        html+="""</table></body></html>"""
        return html

    def create_same_matrix(self,arr,ids):
        '''Given a list of lists with each list consisting of all ids considered same, this function
         generates a matrix '''
        matrix=np.zeros(shape=(ids,ids))
        for subarr in arr:
            for element in subarr:
                matrix[element,subarr]=1
        return matrix

    def create_same_col_matrix(self):
        '''This function will generate same column matrix from available matrices data'''
        all_cols=[]

        for col in range(self.no_of_cols):
            single_col = []
            for subarr in self.data_matrix[:,col]:
                if(subarr is not None):
                    single_col+=subarr
            all_cols.append(single_col)
        return self.create_same_matrix(all_cols,self.idcounter)

    def create_same_row_matrix(self):
        '''This function will generate same row matrix from available matrices data'''
        all_rows=[]

        for row in range(self.no_of_rows):
            single_row=[]
            for subarr in self.data_matrix[row,:]:
                if(subarr is not None):
                    single_row+=subarr
            all_rows.append(single_row)
        return self.create_same_matrix(all_rows,self.idcounter)

    def create_same_cell_matrix(self):
        '''This function will generate same cell matrix from available matrices data'''
        all_cells=[]
        for row in range(self.no_of_rows):
            for col in range(self.no_of_cols):
                if(self.data_matrix[row,col] is not None):
                    all_cells.append(self.data_matrix[row,col])
        return self.create_same_matrix(all_cells,self.idcounter)

    def select_table_category(self):
        '''This function is to make sure that the category of generated table is same as required
        based on selection of table types, border types, row or col spans:
        1. spanflag
        2. tabletype
        3. bordertype
        '''
        #
        tablecategory=1
        if(self.spanflag==False):
            if(self.border_cat==0):
                tablecategory=1
            else:
                tablecategory=2
        else:
            tablecategory=3

        return tablecategory


    def create(self):
        '''This will create the complete table'''
        self.define_col_types()                                             #define the data types for each column
        self.generate_missing_cells()                                       #generate missing cells


        local_span_flag=False                                               #no span initially
        if(self.assigned_category==3):                                      #if assigned category is 3, then it should have spanned rows or columns
            local_span_flag=True
        elif(self.assigned_category==4):                                    #if assigned category is 4, spanning/not spanning doesn't matter
            local_span_flag=random.choices([True,False],weights=[0.5,0.5])[0]   #randomly choose if to span columns and rows for headers or not
        #local_span_flag=True
        if(local_span_flag):
            self.make_header_col_spans()

        html=self.create_html()                                             #create equivalent html

        #create same row, col and cell matrices
        cells_matrix,cols_matrix,rows_matrix=self.create_same_cell_matrix(),\
                                             self.create_same_col_matrix(),\
                                             self.create_same_row_matrix()
        tablecategory=self.select_table_category()                      #select table category of the table
        return cells_matrix,cols_matrix,rows_matrix,self.idcounter,html,tablecategory

