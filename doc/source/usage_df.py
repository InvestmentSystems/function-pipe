

import zipfile
import collections
import os
import webbrowser

import requests
import pandas as pd

from RALib.core import function_pipe as fpn


# source url
URL_NAMES = 'https://www.ssa.gov/oact/babynames/names.zip'
FP_ZIP = '/tmp/names.zip'

class Core:

    def load_data_dict(fp):
        '''Source data from ZIP and load into dictionary of DFs.
        Returns:
            ordered dict of DFs keyed by year
        '''
        # download if not already found
        if not os.path.exists(fp):
            r = requests.get(URL_NAMES)
            with open(fp, 'wb') as f:
                f.write(r.content)

        post = collections.OrderedDict()
        with zipfile.ZipFile(fp) as zf:
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
    dd = Core.load_data_dict(FP_ZIP)
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
        '''Source data from ZIP and load into dictionary of DFs.
        Returns:
            ordered dict of DFs keyed by year
        '''
        # download if not already found
        if not os.path.exists(fp):
            r = requests.get(URL_NAMES)
            with open(fp, 'wb') as f:
                f.write(r.content)

        post = collections.OrderedDict()
        with zipfile.ZipFile(fp) as zf:
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

    @fpn.FunctionNode
    def gender_count_per_year(data_dict):
        records = []
        for year, df in data_dict.items():
            male = df[df['gender'] == 'M']['count'].sum()
            female = df[df['gender'] == 'F']['count'].sum()
            records.append((male, female))
        return pd.DataFrame.from_records(records,
                index=data_dict.keys(), # ordered
                columns=('M', 'F'))

    @fpn.FunctionNode
    def percent(df):
        post = pd.DataFrame(index=df.index)
        sum = df.sum(axis=1)
        for col in df.columns:
            post[col] = df[col] / sum
        return post


    @fpn.FunctionNode
    def year_range(df, start, end):
        return df.loc[start:end]

    @fpn.FunctionNode
    def plot(df, fp='/tmp/plot.png', title=None):
        #print('calling plot', fp)
        if os.path.exists(fp):
            os.remove(fp)
        ax = df.plot(title=title)
        fig = ax.get_figure()
        fig.savefig(fp)
        return fp

    @fpn.FunctionNode
    def open_plot(fp):
        webbrowser.open(fp)


def approach_composition():

    # partional function node arguments
    f = (FN.load_data_dict
        >> FN.gender_count_per_year
        >> FN.year_range.partial(start=1950, end=2000)
        >> FN.percent
        >> FN.plot
        >> FN.open_plot)

    # using operators to scale percent
    #f = (FN.load_data_dict
        #>> FN.gender_count_per_year
        #>> FN.year_range.partial(start=1950, end=2000)
        #>> (FN.percent * 100)
        #>> FN.plot
        #>> FN.open_plot)

    f(FP_ZIP)

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

    fpn.run(f, FP_ZIP)




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

    f = (PN2.load_data_dict(FP_ZIP) | PN2.gender_count_per_year
        | PN2.percent * 100 | PN2.year_range(1900, 2000)
        | PN2.plot | PN2.open_plot)

    # with store and recall to do multiple operations

    f = (PN2.load_data_dict(FP_ZIP) | PN2.gender_count_per_year
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



class PN4:
    # refactor methods to take to take args
    # add method that gets


    class PNI(fpn.PipeNodeInput):

        URL_NAMES = 'https://www.ssa.gov/oact/babynames/names.zip'

        @classmethod
        def load_data_dict(cls, fp):

            if not os.path.exists(fp):
                r = requests.get(cls.URL_NAMES)
                with open(fp, 'wb') as f:
                    f.write(r.content)

            post = collections.OrderedDict()
            with zipfile.ZipFile(fp) as zf:
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

        def __init__(self, output_dir):
            super().__init__()
            self.output_dir = output_dir
            fp_zip = os.path.join(output_dir, 'names.zip')
            self.data_dict = self.load_data_dict(fp_zip)


    # new function that is more general than gender_count_per_year
    @fpn.pipe_node_factory
    def name_count_per_year(name_match, **kwargs):
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
        df = kwargs[fpn.PREDECESSOR_RETURN]
        post = pd.DataFrame(index=df.index)
        sum = df.sum(axis=1)
        for col in df.columns:
            post[col] = df[col] / sum
        return post

    @fpn.pipe_node_factory
    def year_range(start, end, **kwargs):
        return kwargs[fpn.PREDECESSOR_RETURN].loc[start:end]

    @fpn.pipe_node_factory
    def plot(file_name, title=None, **kwargs): # now we can pass a file name
        pni = kwargs[fpn.PN_INPUT]
        df = kwargs[fpn.PREDECESSOR_RETURN]
        fp = os.path.join(pni.output_dir, file_name)
        ax = df.plot(title=title)
        ax.get_figure().savefig(fp)
        print(fp)
        return fp

    @fpn.pipe_node
    def open_plot(**kwargs):
        webbrowser.open(kwargs[fpn.PREDECESSOR_RETURN])


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


    #@fpn.pipe_node
    #def write_xlsx(**kwargs):
        #pni = kwargs[fpn.PN_INPUT]
        #xlsx_fp = os.path.join(pni.output_dir, 'output.xlsx')
        #xlsx = pd.ExcelWriter(xlsx_fp)
        #for k, df in pni.store_items():
            #df.to_excel(xlsx, k)
        #xlsx.save()
        #return xlsx_fp

def approach_pipe_4a():

    f = (PN4.name_count_per_year(lambda n: n.lower().startswith('lesl'))
        | PN4.percent | PN4.plot('lesl.png') | PN4.open_plot
        | PN4.name_count_per_year(lambda n: n.lower().startswith('dana'))
        | PN4.percent | PN4.plot('dana.png') | PN4.open_plot
        )

    f[PN4.PNI('/tmp')] # use our derived PipeNodeInput



def approach_pipe_4b():

    a = (PN4.name_count_per_year(lambda n: n.lower().startswith('lesl'))
            | PN4.percent | fpn.store('lesl'))

    b = (PN4.name_count_per_year(lambda n: n.lower().startswith('dana'))
            | PN4.percent | fpn.store('dana'))

    f = (PN4.merge_gender_data(lesl=a, dana=b)
            | PN4.year_range(1920, 2000)
            | fpn.store('merged') * 100
            | PN4.plot('gender.png')
            | PN4.open_plot)

    pni = PN4.PNI('/tmp')
    f[pni]


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


