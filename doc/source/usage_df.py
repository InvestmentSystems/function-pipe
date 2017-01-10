

import zipfile
import collections
import os
import shutil

from functools import partial

import pandas as pd

from RALib.core import function_pipe as fpn


# source url
url = 'https://www.ssa.gov/oact/babynames/names.zip'
zip_fp = '/tmp/names.zip'

class Core:

    def load_data_dict(fp):
        '''Source data from ZIP and load into dictionary of DFs.
        Returns:
            ordered dict of DFs keyed by year
        '''
        post = collections.OrderedDict()
        with zipfile.ZipFile(zip_fp) as zf:
            #import ipdb; ipdb.set_trace()
            # get ZipInfo instances
            for zi in sorted(zf.infolist(), key=lambda zi: zi.filename):
                fn = zi.filename
                if fn.startswith('yob'):
                    year = int(fn[3:7])
                    df = pd.read_csv(
                            zf.open(zi),
                            header=None,
                            names=('name', 'gender', 'count'))
                    df['year'] = year
                    post[year] = df
        return post

    def gender_count_per_year(data_dict):
        records = []
        for year, df in data_dict.items():
            male = df[df['gender'] == 'M']['count'].sum()
            female = df[df['gender'] == 'F']['count'].sum()
            records.append((male, female))
        return pd.DataFrame.from_records(records,
                index=data_dict.keys(), # ordered
                columns=('M', 'F'))

    def percent(df):
        post = pd.DataFrame(index=df.index)
        sum = df.sum(axis=1)
        for col in df.columns:
            post[col] = df[col] / sum
        return post

    def year_range(df, start, end):
        return df.loc[start:end]

    def plot(df, fp='/tmp/plot.png', title=None):
        #print('calling plot', fp)
        if os.path.exists(fp):
            os.remove(fp)
        ax = df.plot(title=title)
        fig = ax.get_figure()
        fig.savefig(fp)
        return fp

    def open_plot(fp):
        print('calling open plot')
        os.system('eog ' + fp)


#-------------------------------------------------------------------------------
# approach 1: call lots of functions wiht lots of statements
def approach_statement():
    dd = Core.load_data_dict(zip_fp)
    #ddf = Core.data_df(dd)
    g_count = Core.gender_count_per_year(dd)
    g_percent = Core.percent(g_count)
    g_sub = Core.year_range(g_percent, 1950, 2000)
    fp = Core.plot(g_sub)
    Core.open_plot(fp)


#-------------------------------------------------------------------------------

class FN:

    @fpn.FunctionNode
    def load_data_dict(fp):
        return Core.load_data_dict(fp)

    @fpn.FunctionNode
    def gender_count_per_year(data_dict):
        return Core.gender_count_per_year(data_dict)

    @fpn.FunctionNode
    def percent(data_dict):
        return Core.percent(data_dict)

    @fpn.FunctionNode
    def year_range(df, start, end):
        return Core.year_range(df, start, end)

    @fpn.FunctionNode
    def plot(df):
        return Core.plot(df)

    @fpn.FunctionNode
    def open_plot(df):
        return Core.open_plot(df)

    @fpn.FunctionNode
    def year_range(df, start, end):
        return Core.year_range(df, start, end)

def approach_composition():

    # partional function node arguments
    f = (FN.load_data_dict
        >> FN.gender_count_per_year
        >> FN.year_range.partial(start=1950, end=2000)
        >> FN.percent
        >> FN.plot
        >> FN.open_plot)

    # using operators to scale percent
    f = (FN.load_data_dict
        >> FN.gender_count_per_year
        >> FN.year_range.partial(start=1950, end=2000)
        >> (FN.percent * 100)
        >> FN.plot
        >> FN.open_plot)

    f(zip_fp)

    # how do we plot more than one thing without resourcing data


#-------------------------------------------------------------------------------

class PN1:

    @fpn.pipe_node
    @fpn.pipe_kwarg_bind(fpn.PN_INPUT)
    def load_data_dict(fp):
        return Core.load_data_dict(fp)

    @fpn.pipe_node
    @fpn.pipe_kwarg_bind(fpn.PREDECESSOR_RETURN)
    def gender_count_per_year(data_dict):
        return Core.gender_count_per_year(data_dict)

    @fpn.pipe_node
    @fpn.pipe_kwarg_bind(fpn.PREDECESSOR_RETURN)
    def percent(df):
        return Core.percent(df)

    @fpn.pipe_node_factory
    @fpn.pipe_kwarg_bind(fpn.PREDECESSOR_RETURN)
    def year_range(df, start, end):
        return Core.year_range(df, start, end)

    @fpn.pipe_node
    @fpn.pipe_kwarg_bind(fpn.PREDECESSOR_RETURN)
    def plot(df):
        return Core.plot(df)

    @fpn.pipe_node
    @fpn.pipe_kwarg_bind(fpn.PREDECESSOR_RETURN)
    def open_plot(df):
        return Core.open_plot(df)


def approach_pipe_1():
    f = (PN1.load_data_dict | PN1.gender_count_per_year
        | PN1.percent | PN1.year_range(1900, 2000)
        | PN1.plot | PN1.open_plot)

    # with operator
    f = (PN1.load_data_dict | PN1.gender_count_per_year
        | PN1.percent * 100 | PN1.year_range(1900, 2000)
        | PN1.plot | PN1.open_plot)

    fpn.run(f, zip_fp)




#-------------------------------------------------------------------------------
# alternate approach where component functions take kwargs; only exposed args are those for pipe factories; use PipNodeInput as input

class PN2:

    @fpn.pipe_node_factory
    def load_data_dict(fp, **kwargs):
        return Core.load_data_dict(fp)

    @fpn.pipe_node
    def gender_count_per_year(**kwargs):
        return Core.gender_count_per_year(kwargs[fpn.PREDECESSOR_RETURN])

    @fpn.pipe_node
    def percent(**kwargs):
        return Core.percent(kwargs[fpn.PREDECESSOR_RETURN])

    @fpn.pipe_node_factory
    def year_range(start, end, **kwargs):
        return Core.year_range(kwargs[fpn.PREDECESSOR_RETURN], start, end)

    @fpn.pipe_node
    def plot(**kwargs):
        return Core.plot(kwargs[fpn.PREDECESSOR_RETURN])

    @fpn.pipe_node
    def open_plot(**kwargs):
        return Core.open_plot(kwargs[fpn.PREDECESSOR_RETURN])


def approach_pipe_2():

    f = (PN2.load_data_dict(zip_fp) | PN2.gender_count_per_year
        | PN2.percent * 100 | PN2.year_range(1900, 2000)
        | PN2.plot | PN2.open_plot)

    # with store and recall to do multiple operations

    f = (PN2.load_data_dict(zip_fp) | PN2.gender_count_per_year
        | PN2.percent * 100
        | fpn.store('gpcent')
        | PN2.year_range(1900, 2000) | PN2.plot | PN2.open_plot |
        fpn.recall('gpcent')
        | PN2.year_range(2001, 2015) | PN2.plot | PN2.open_plot
        )

    #fpn.run(f) # this implicitly passes a PipeNodeInput
    f(pn_input=fpn.PipeNodeInput())

#-------------------------------------------------------------------------------
# use pipe node input to distribute data dict, also output directory

class PN3:

    class PNI(fpn.PipeNodeInput):
        zip_fp = zip_fp

        def __init__(self):
            super().__init__()
            self.data_dict = Core.load_data_dict(self.zip_fp)
            self.output_dir = '/tmp'

    @fpn.pipe_node
    def gender_count_per_year(**kwargs):
        pni = kwargs[fpn.PN_INPUT]
        return Core.gender_count_per_year(pni.data_dict)

    @fpn.pipe_node
    def percent(**kwargs):
        return Core.percent(kwargs[fpn.PREDECESSOR_RETURN])

    @fpn.pipe_node_factory
    def year_range(start, end, **kwargs):
        return Core.year_range(kwargs[fpn.PREDECESSOR_RETURN], start, end)

    @fpn.pipe_node_factory # now a factory
    def plot(file_name, **kwargs): # now we can pass a file name
        pni = kwargs[fpn.PN_INPUT]
        fp = os.path.join(pni.output_dir, file_name)
        # now passing file path
        return Core.plot(kwargs[fpn.PREDECESSOR_RETURN], fp)

    @fpn.pipe_node
    def open_plot(**kwargs):
        return Core.open_plot(kwargs[fpn.PREDECESSOR_RETURN])

def approach_pipe_3a():

    f = (PN3.gender_count_per_year
        | fpn.store('gcount') # store count and percent
        | PN3.percent * 100
        | fpn.store('gpcent')
        | PN3.year_range(1900, 2000) | PN3.plot('20c.png') | PN3.open_plot |
        fpn.recall('gpcent')
        | PN3.year_range(2001, 2015) | PN3.plot('21c.png') | PN3.open_plot
        )

    pni = PN3.PNI()
    f(pn_input=pni) # use our derived PipeNodeInput

    # we can get the things we stored after
    for k, v in pni.store_items():
        print(k)
        print(v.dtypes)

def approach_pipe_3b():
    # breaking up the single expression into multiple expressions

    a = (PN3.gender_count_per_year
        | fpn.store('gcount') # store count and percent
        | PN3.percent * 100
        | fpn.store('gpcent'))

    b = (fpn.recall('gpcent') | PN3.year_range(1900, 2000)
        | PN3.plot('20c.png') | PN3.open_plot)

    c = (fpn.recall('gpcent') | PN3.year_range(2001, 2015)
        | PN3.plot('21c.png') | PN3.open_plot)

    #f = a | b | c # sorry, cant do this; these are process, not expression, FNs

    # can use call(); this itself is an PN that can go on to do other opperations
    f = fpn.call(a, b, c) | fpn.pipe_node(lambda **kwargs: print('done'))

    pni = PN3.PNI()
    f(pn_input=pni) # use our derived PipeNodeInput

    # we can get the things we stored after
    for k, v in pni.store_items():
        print(k)
        print(v.dtypes)



class PN4:
    # refactor methods to take to take args
    # add method that gets

    class PNI(fpn.PipeNodeInput):
        zip_fp = zip_fp

        def __init__(self):
            super().__init__()
            self.data_dict = Core.load_data_dict(self.zip_fp)
            self.output_dir = '/tmp'


    # new function that is more general than gender_count_per_year
    @fpn.pipe_node_factory
    def name_count_per_year(name_match, **kwargs):
        '''
        Args:
            name_match: function that returns a Boolean if it matches the targetted name
        '''
        pni = kwargs[fpn.PN_INPUT]
        records = []
        for year, df in pni.data_dict.items():
            counts = collections.OrderedDict()
            sel_name = df['name'].apply(name_match)
            for gender in ('M', 'F'):
                sel_gender = (df['gender'] == gender) & sel_name
                counts[gender] = df[sel_gender]['count'].sum()
            records.append(tuple(counts.values()))

        return pd.DataFrame.from_records(records,
                index=pni.data_dict.keys(), # ordered
                columns=('M', 'F'))

    @fpn.pipe_node
    def percent(**kwargs):
        return Core.percent(kwargs[fpn.PREDECESSOR_RETURN])

    @fpn.pipe_node_factory
    def year_range(start, end, **kwargs):
        return Core.year_range(kwargs[fpn.PREDECESSOR_RETURN], start, end)

    @fpn.pipe_node_factory # now a factory
    def plot(file_name, title=None, **kwargs): # now we can pass a file name
        pni = kwargs[fpn.PN_INPUT]
        fp = os.path.join(pni.output_dir, file_name)
        # now passing file path
        return Core.plot(kwargs[fpn.PREDECESSOR_RETURN], fp, title)

    @fpn.pipe_node
    def open_plot(**kwargs):
        return Core.open_plot(kwargs[fpn.PREDECESSOR_RETURN])

    @fpn.pipe_node_factory
    def merge_gender_data(**kwargs):
        pni = kwargs[fpn.PN_INPUT]
        # get index from source data dict
        df = pd.DataFrame(index=pni.data_dict.keys())
        for k, v in kwargs.items():
            if k not in fpn.PIPE_NODE_KWARGS:
                for gender in ('M', 'F'):
                    df[k + '_' + gender] = v[gender]
        return df

def approach_pipe_4a():

    a = (PN4.name_count_per_year(lambda n: n.lower().startswith('lesl'))
        | PN4.percent)

    b = (PN4.name_count_per_year(lambda n: n.lower().startswith('dana'))
        | PN4.percent)

    c = (PN4.name_count_per_year(lambda n: n.lower().startswith('sydney'))
        | PN4.percent)

    f = (PN4.merge_gender_data(lesl=a, dana=b, sydney=c)
        | PN3.year_range(1900, 2016) | PN3.plot('gender.png') | PN3.open_plot)

    pni = PN4.PNI()
    f(pn_input=pni) # use our derived PipeNodeInput




def approach_pipe_4b():
    # here we put creation of PNs in a loop, and store all the raw counts

    names = (('lesl', lambda n: n.lower().startswith('lesl')),
            ('dana', lambda n: n.lower().startswith('dana')),
            ('sydney', lambda n: n.lower().startswith('sydney')),
            #('paris', lambda n: n.lower().startswith('paris')),
            )

    parts = {}
    for name, selector in names:
        q = (PN4.name_count_per_year(selector)
            | fpn.store(name + '_count')
            | PN4.percent * 100
            | fpn.store(name + '_percent'))
        parts[name] = q

    f = (PN4.merge_gender_data(**parts)
            | fpn.store('merged')
            | PN4.year_range(1900, 2016)
            | PN4.plot('gender.png') | PN4.open_plot)

    pni = PN4.PNI()
    f(pn_input=pni)

    xlsx_fp = os.path.join(pni.output_dir, 'output.xlsx')
    xlsx = pd.ExcelWriter(xlsx_fp)
    for k, df in pni.store_items():
        df.to_excel(xlsx, k)
    xlsx.save()

    os.system('libreoffice --calc ' + xlsx_fp)



if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        func = locals().get(sys.argv[1], None)
        if func:
            print(func)
            func()

    #pass
    ##approach_composition()
    ##approach_pipe_1()
    ##approach_pipe_2()
    ##approach_pipe_3a()
    ##approach_pipe_3b()
    ##approach_pipe_3b()
    #approach_pipe_4b()

