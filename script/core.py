#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
import numpy as np
import pandas as pd
import scanpy as sc 
import anndata as ad
import math
import operator
import collections
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from networkx.drawing.nx_agraph import graphviz_layout


import warnings                 #忽略warning
warnings.filterwarnings('ignore')


class IdtiData(object):                 #创建IdtiData类   类定义不能为空，但是如果您处于某种原因写了无内容的类定义语句，请使用 pass 语句来避免错误。

    timepoint_scdata_dict = collections.OrderedDict()                   #有序字典存放按时间分组的anndata对象
    __timepoint_scdata_num=1
    nodes=[]  
    edges=[]
    numofcell=[]
    dic_stage={}
    stage_num=int()
    stage_name=[]


    def __init__(self,data):             
        self.data=data  

        # Step0:output#
        self.__Output_Dir="./"
        self.__Output_Name="IDTI"

        #Step1:cell_cluster#                    #添加类属性
        self.__CellClusterParam_PCAn=10
        self.__CellClusterParam_k=15
        self.__CellClusterParam_Resolution=0.1    

        #Step2:data_processing#
        # gene_and _cell _filter#
        self.__Switch_DeadCellFilter=False
        self.__Threshold_MitoPercent=0.2
            
        self.__Switch_LowGeneCellsFilter=True
        self.__Threshold_LowGeneNum=200

        self.__Switch_LowCellGenesFilter=True
        self.__Threshold_LowCellNum=3

        #normalize and log_transform#
        self.__Switch_Normalize=True
        self.__Threshold_NormalizeBase=1e4              
        self.__Switch_LogTransform=True 

        #highly_variable_genes and scale#
        self.__Switch_HightlyVariableGenes=False
        self.__Threshold_HightlyVariableGenes=1000
        self.__Switch_ScaleData_MinMax=True

        #Step3:data partition, calculation and visualization#
        self.__Threshold_GeneExpress=0                  #基因进一步过滤条件
        self.__Threshold_NameOfStage='timepoints'                    #添加数据细胞时间标签的索引
        self.__Threshold_NameOfCellState='cell_type'                   #添加数据细胞类型标签的索引
        self.__Threshold_NumOfStage=['T1','T2','T3']
        self.__Threshold_MinCellNumofStates=0

 
    def __str__(self):
        s="" 
        s+=f"\n#Step0:output# \n"
        s+=f"Output_Dir={self.__Output_Dir}\n"
        s+=f"Output_Name={self.__Output_Name}\n"         
            
        s+=f"\n#Step1:cell_cluster# \n"
        s+=f"CellClusterParam_PCAn={self.__CellClusterParam_PCAn}\n"
        s+=f"CellClusterParam_k={self.__CellClusterParam_k}\n"
        s+=f"CellClusterParam_Resolution={self.__CellClusterParam_Resolution}\n"

        s+=f"\n#Step2:gene_and _cell _filter#\n"
        s+=f"Switch_DeadCellFilter={self.__Switch_DeadCellFilter}\n"
        s+=f"Threshold_MitoPercent={self.__Threshold_MitoPercent}\n"

        s+=f"Switch_LowCellGenesFilter={self.__Switch_LowCellGenesFilter}\n"
        s+=f"Threshold_LowCellNum={self.__Threshold_LowCellNum}\n"

        s+=f"Switch_LowGeneCellsFilter={self.__Switch_LowGeneCellsFilter}\n"
        s+=f"Threshold_LowGeneNum={self.__Threshold_LowGeneNum}\n"

        s+=f"\n#Step3:normalize and log_transform#\n"
        s+=f"Switch_Normalize={self.__Switch_Normalize}\n"
        s+=f"Threshold_NormalizeBase={self.__Threshold_NormalizeBase}\n"
        s+=f"Switch_LogTransform={self.__Switch_LogTransform}\n"

        s+=f"\n#Step5:highly_variable_genes and scale#\n"
        s+=f"Switch_HightlyVariableGenes={self.__Switch_HightlyVariableGenes}\n"
        s+=f"Threshold_HightlyVariableGenes={self.__Threshold_HightlyVariableGenes}\n"
        s+=f"Switch_ScaleData={self.__Switch_ScaleData}\n"

        s+=f"\n#Step6:calculate_id and plot_graph#\n"
        s+=f"Threshold_GeneExpress={self.__Threshold_GeneExpress}\n"
        s+=f"Threshold_NameOfStage={self.__Threshold_NameOfStage}\n"
        s+=f"Threshold_NameOfCellState={self.__Threshold_NameOfCellState}\n"
        s+=f"Threshold_NumOfStage={self.__Threshold_NumOfStage}\n"
        s+=f"Threshold_MinCellNumofStates={self.__Threshold_MinCellNumofStates}\n"
            
        return s


    def __repr__(self):                 #__repr__ = __str__
        return self.__str__()


    def add_new_timepoint_scdata(self,timepoint_scdata,timepoint_scdata_cluster=None,**kwargs):
        Threshold_GeneExpress=self.__Threshold_GeneExpress=kwargs.setdefault("Threshold_GeneExpress",self.__Threshold_GeneExpress)
        #print(f'filtered genes with st >= {Threshold_GeneExpress} ...')
        data=pd.DataFrame(timepoint_scdata)
        ind=(data.std()>=Threshold_GeneExpress).to_list()
        data=data.loc[:,ind]
        self.timepoint_scdata_dict[self.__timepoint_scdata_num]=ad.AnnData(data)
        if timepoint_scdata_cluster != None:
            self.timepoint_scdata_dict[self.__timepoint_scdata_num].obs["scdata_cluster"]=[f"timepoint{self.__timepoint_scdata_num}_{c}" for c in list(timepoint_scdata_cluster)]   #节点标签
            # self.timepoint_scdata_dict[self.__timepoint_scdata_num].obs["scdata_cluster"]=[f"{c}" for c in list(timepoint_scdata_cluster)]
            self.timepoint_scdata_dict[self.__timepoint_scdata_num].uns["cluster_flag"]=True
        else:
            self.timepoint_scdata_dict[self.__timepoint_scdata_num].obs["scdata_cluster"]=[0]*self.timepoint_scdata_dict[self.__timepoint_scdata_num].n_obs
            self.timepoint_scdata_dict[self.__timepoint_scdata_num].uns["cluster_flag"]=False
        self.__timepoint_scdata_num+=1


    def __create_folder(self):
        Output_Dir = self.__Output_Dir
        Output_Name = self.__Output_Name
        if not os.path.exists(Output_Dir):
            raise ValueError(f'{Output_Dir} : No such directory')
        elif not os.path.exists(Output_Dir+Output_Name):
            os.makedirs(Output_Dir+Output_Name)
        else:
            print(f"The result folder {Output_Dir+Output_Name} exists! IDTI overwrite it.")

        if not os.path.exists(Output_Dir+Output_Name+"/"+"SupplementaryResults"):
            os.makedirs(Output_Dir+Output_Name+"/"+"SupplementaryResults")


    def cell_clusters(self,**kwargs):

        self.__create_folder()

        CellClusterParam_PCAn = self.__CellClusterParam_PCAn = kwargs.setdefault('CellClusterParam_PCAn', self.__CellClusterParam_PCAn)
        CellClusterParam_k = self.__CellClusterParam_k = kwargs.setdefault('CellClusterParam_k', self.__CellClusterParam_k)
        CellClusterParam_Resolution = self.__CellClusterParam_Resolution=kwargs.setdefault("CellClusterParam_Resolution",self.__CellClusterParam_Resolution)
        normalize_flag = self.__Switch_Normalize=kwargs.setdefault("Switch_Normalize",self.__Switch_Normalize)
        log_flag = self.__Switch_LogTransform=kwargs.setdefault("Switch_LogTransform",self.__Switch_LogTransform)
        
        for (timepoint,adata) in self.timepoint_scdata_dict.items():
            print(f"timepoint:{timepoint}")
            if adata.uns['cluster_flag'] :
                print(f"clusters have been given")
            else:
                adata_copy=adata.copy()
                adata_copy.obs_names_make_unique()
                adata_copy.var_names_make_unique()
                # MT pct
                mito_genes = adata_copy.var_names.str.startswith('MT-')
                adata_copy.obs['percent_mito'] = np.sum(adata_copy[:, mito_genes].X, axis=1) / np.sum(adata_copy.X, axis=1)
                adata_copy.obs['n_counts'] = adata_copy.X.sum(axis=1)
                # normalize log-transform
                if normalize_flag:
                    sc.pp.normalize_per_cell(adata_copy, counts_per_cell_after=1e4)
                if log_flag:
                    sc.pp.log1p(adata_copy)
                # high variable genes
                sc.pp.highly_variable_genes(adata_copy, min_mean=0.0125, max_mean=3, min_disp=0.5)
                adata_high = adata_copy[:, adata_copy.var['highly_variable']]
                # linear regression
                sc.pp.regress_out(adata_high, ['n_counts', 'percent_mito'])
                sc.pp.scale(adata_high, max_value=10)
                # pca
                sc.tl.pca(adata_high, n_comps=CellClusterParam_PCAn, svd_solver='arpack')
                # knn
                sc.pp.neighbors(adata_high, n_neighbors=CellClusterParam_k, n_pcs=CellClusterParam_PCAn)
                sc.tl.louvain(adata_high, resolution=CellClusterParam_Resolution)
                adata_high.obs["louvain"]=[str(int(i)+1) for i in adata_high.obs["louvain"]]
                sc.tl.umap(adata_high)
                umap_cord=pd.DataFrame(adata_high.obsm["X_umap"])
                umap_cord.index=adata_high.obs.index
                umap_cord["louvain"]=adata_high.obs["louvain"]
                umap_cord.columns=["x","y","louvain"]

                fig=sc.pl.umap(adata_high, color='louvain',return_fig=True)
                plt.show(block=False)
                plt.pause(1.0)
                plt.close()             
                adata.obs["scdata_cluster"]=[f"timepoint{timepoint}_cluster{int(c)}" for c in adata_high.obs["louvain"].tolist()]
                adata.uns["cluster_flag"]=True                #添加细胞状态标签
            # 按字符串排序
            node_cluster=adata.obs["scdata_cluster"]
            #node_cluster.index=adata.obs["cell_id"]
            cluster_set=node_cluster.unique().tolist()
            cluster_set.sort()
            adata.uns["cluster_set"]=cluster_set
            adata.uns["cluster_counts"]=node_cluster.value_counts()


    def filter_dead_cell(self,**kwargs):
        Threshold_MitoPercent=self.__Threshold_MitoPercent=kwargs.setdefault("Threshold_MitoPercent",self.__Threshold_MitoPercent)
        self.data.var['mt'] = self.data.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        self.data.obs['percent_mito'] = np.sum(self.data[:, self.data.var['mt']].X, axis=1) / np.sum(self.data.X, axis=1)
        self.data.obs['n_counts'] = self.data.X.sum(axis=1)
        raw_cell_num=self.data.n_obs
        self.data=self.data[self.data.obs['percent_mito']<Threshold_MitoPercent,:]
        filter_cell_num=self.data.n_obs
        print(f'filtered out {raw_cell_num-filter_cell_num} cells that are detected in more than {Threshold_MitoPercent} mito percent')
        print()


    def filter_lowgene_cells(self,**kwargs):
        Threshold_LowGeneNum=self.__Threshold_LowGeneNum=kwargs.setdefault("Threshold_LowGeneNum",self.__Threshold_LowGeneNum)
        raw_cell_num=self.data.n_obs
        sc.pp.filter_cells(self.data, min_genes=Threshold_LowGeneNum)
        filter_cell_num=self.data.n_obs
        print(f'filtered out {raw_cell_num-filter_cell_num} cells that are detected in less than {Threshold_LowGeneNum} genes')
        print()
        
    
    def filter_lowcell_genes(self,**kwargs):
        Threshold_LowCellNum=self.__Threshold_LowCellNum=kwargs.setdefault("Threshold_LowCellNum",self.__Threshold_LowCellNum)
        raw_gene_num=self.data.n_vars
        sc.pp.filter_genes(self.data, min_cells=Threshold_LowCellNum)
        filter_gene_num=self.data.n_vars
        print(f'filtered out {raw_gene_num-filter_gene_num} genes that are detected in less than {Threshold_LowCellNum} cells')
        print()
  
    
    def normalize_data(self,**kwargs):
        Threshold_NormalizeBase=self.__Threshold_NormalizeBase=kwargs.setdefault("Threshold_NormalizeBase",self.__Threshold_NormalizeBase)
        sc.pp.normalize_total(self.data,target_sum=Threshold_NormalizeBase)
        print(f'Normalize data to {Threshold_NormalizeBase} count ...')
        print()


    def log_transform(self):
        sc.pp.log1p(self.data)
        self.data.raw = self.data
        print(f'transfor data...')
        print()

 
    def highly_variable_genes(self,**kwargs):
        Threshold_HightlyVariableGenes=self.__Threshold_HightlyVariableGenes=kwargs.setdefault("Threshold_HightlyVariableGenes",self.__Threshold_HightlyVariableGenes)
        sc.pp.highly_variable_genes(self.data, flavor="seurat", n_top_genes=Threshold_HightlyVariableGenes) 
        self.data =self.data[:, self.data.var.highly_variable]
        print(f'selected highly variable genes {Threshold_HightlyVariableGenes} ...')
        print()


    def scaled_data_MinMaxScale(self):
        scaledata = MinMaxScaler(feature_range=(0, 1))                      #最小最大归一化
        self.data.X=scaledata.fit_transform(self.data.X)
        print(f'scale data with the method of MinMaxScaler...')
        print()

    
    def group_stage(self,**kwargs):
        Threshold_NameOfStage=self.__Threshold_NameOfStage=kwargs.setdefault("Threshold_NameOfStage",self.__Threshold_NameOfStage)
        Threshold_NameOfCellState=self.__Threshold_NameOfCellState=kwargs.setdefault("Threshold_NameOfCellState",self.__Threshold_NameOfCellState)
        set_stage=cdata.data.obs[Threshold_NameOfStage]
        for stage in iter(sorted(list(set(set_stage.value_counts().index)))):                #set无序   
            self.dic_stage[stage]=cdata.data[set_stage==stage]
            self.add_new_timepoint_scdata(pd.DataFrame(cdata.data[set_stage==stage].X),(cdata.data[set_stage==stage].obs[Threshold_NameOfCellState].tolist())) 
        self.stage_num=(len(sorted(list(set(set_stage.value_counts().index)))))
        print(f'basic partitions with timepoints...')
        print()


    def name_stage(self,**kwargs):
        Threshold_NumOfStage=self.__Threshold_NumOfStage=kwargs.setdefault("Threshold_NumOfStage",self.__Threshold_NumOfStage)
        if Threshold_NumOfStage:
            self.stage_name=list(reversed(Threshold_NumOfStage))
        else:
            for i in range(self.stage_num):
                self.stage_name.append("T"+str(i+1))
            self.stage_name=list(reversed(self.stage_name))
        return(self.stage_name)


    def D(self,X):
        N = 0
        ans = 0
        length = len(X)
        for x in X:       
            if(x==0):        #忽略log0值
                continue
            else:
                n = x   
                N = N + n
                ans += n * math.log(n)
            Dx = N * math.log(N) - ans
        return Dx


    def Id(self, X, Y):
        Dx = self.D(X)
        Dy = self.D(Y)
        # XY = np.sum([X, Y], axis=0).tolist()
        XY = np.sum([X, Y], axis=0)
        Dxy = self.D(XY)
        id = Dxy - Dx - Dy
        return id


    def draw_nodes(self,**kwargs):
        dic_all={}

        Threshold_MinCellNumofStates=self.__Threshold_MinCellNumofStates=kwargs.setdefault("Threshold_MinCellNumofStates",self.__Threshold_MinCellNumofStates)
        for (timepoint,adata) in cdata.timepoint_scdata_dict.items():
            pd1=pd.concat([pd.DataFrame(adata.obs.scdata_cluster),pd.DataFrame(adata.X, columns=adata.var.index, index=adata.obs.index)],axis=1)
            pd2 = pd1.groupby('scdata_cluster').sum()
            count = Counter(adata.obs['scdata_cluster'].tolist())
            drop_state = [k for k,v in count.items() if v <= Threshold_MinCellNumofStates]
            pd2=pd2.drop(drop_state, axis=0)   
            dic_all[timepoint]=pd2

            dic1={}
            dic2={}
            if(timepoint==1):
                for i in range(len(dic_all[timepoint])):
                    dic1[dic_all[timepoint].index[i]]=0
                self.nodes.extend(dic1)
                continue
            else:
                for i in range(len(dic_all[timepoint])):
                    dic = {}
                    choice = float("inf")
                    for j in range(len(dic_all[timepoint-1])):
                        difference = self.Id(dic_all[timepoint].values.tolist()[i],dic_all[timepoint-1].values.tolist()[j])
                        edge = (dic_all[timepoint-1].index[j],dic_all[timepoint].index[i])
                        dic[difference] = edge
                        choice = min(difference, choice)
                        dic2[dic_all[timepoint].index[i]]=choice   
            self.nodes.extend(dict(sorted(dic2.items(),key=operator.itemgetter(1))))
        return(self.nodes)
            

    def draw_edges(self,**kwargs):
        dic_all={}

        Threshold_MinCellNumofStates=self.__Threshold_MinCellNumofStates=kwargs.setdefault("Threshold_MinCellNumofStates",self.__Threshold_MinCellNumofStates)
        for (timepoint,adata) in cdata.timepoint_scdata_dict.items():
            pd1=pd.concat([pd.DataFrame(adata.obs.scdata_cluster),pd.DataFrame(adata.X, columns=adata.var.index, index=adata.obs.index)],axis=1)
            pd2 = pd1.groupby('scdata_cluster').sum()
            count = Counter(adata.obs['scdata_cluster'].tolist())
            drop_state = [k for k,v in count.items() if v <= Threshold_MinCellNumofStates]
            pd2=pd2.drop(drop_state, axis=0)   
            dic_all[timepoint]=pd2

            if(timepoint==1):
                continue
            else:
                for i in range(len(dic_all[timepoint])):
                    dic = {}
                    choice = float("inf")
                    for j in range(len(dic_all[timepoint-1])):
                        difference = self.Id(dic_all[timepoint].values.tolist()[i],dic_all[timepoint-1].values.tolist()[j])
                        edge = (dic_all[timepoint-1].index[j],dic_all[timepoint].index[i])
                        dic[difference] = edge
                        choice = min(difference, choice)
                    self.edges.append(dic[choice])    
        return(self.edges)


    def adjust_statenum(self,**kwargs):
        Threshold_MinCellNumofStates=self.__Threshold_MinCellNumofStates=kwargs.setdefault("Threshold_MinCellNumofStates",self.__Threshold_MinCellNumofStates)
        for (timepoint,adata) in cdata.timepoint_scdata_dict.items():
            list=adata.obs.scdata_cluster.tolist()
            df = pd.DataFrame.from_dict(Counter(list), orient='index').reset_index()
            df.columns = ['key', 'cnts']
            self.numofcell = self.numofcell+df.cnts.tolist()
        self.numofcell = np.array(self.numofcell)    
        self.numofcell = self.numofcell[self.numofcell>Threshold_MinCellNumofStates].tolist()   
        return(self.numofcell)

    
    def plot_umap(self,**kwargs):
        print(f'calculate ID and output results...')

        
        sc.pp.pca(adata)  
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        fig = sc.pl.umap(adata, title='', color="cell_type",use_raw=False,size=80,frameon=False, return_fig=True)

        umap_cord=pd.DataFrame(adata.obsm["X_umap"])
        umap_cord.index=adata.obs.index
        umap_cord["cell_type"]=adata.obs[self.__Threshold_NameOfCellState]
        umap_cord.columns=["x","y","cell_type"]
        clusters=umap_cord.groupby("cell_type").agg('mean')
        def my_point(a,b):
            return (a,b)
        clusters['points']=clusters.groupby("cell_type").agg('mean').apply(lambda row:my_point(row['x'],row['y']),axis=1)

        G = nx.DiGraph()    #有向图    
        node_list=[]
        for i,nodes in enumerate(self.draw_edges()):
            tuple=()
            for j, nodes in enumerate(nodes):
                nodes=("".join(nodes).split('_',1))[1]
                tuple=tuple+(nodes,)
            node_list.append(tuple)
    
        G.add_edges_from(node_list)                #  self.draw_edges()                   #  
        pos=clusters['points'].to_dict()

        nx.draw_networkx_nodes(G, pos, node_size=30, node_color='k',node_shape='o')
        curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
        straight_edges = list(set(G.edges()) - set(curved_edges))
        nx.draw_networkx_edges(G, pos, width=1.2, edgelist=straight_edges)
        nx.draw_networkx_edges(G, pos, width=1.2, edgelist=curved_edges, connectionstyle='arc3, rad = 0.25')
        plt.show()
        plt.close()

        output_path=self.__Output_Dir+self.__Output_Name+"/"
        output_name=self.__Output_Name
        fig.savefig(output_path+f"{output_name}_umap.tif", bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"}, dpi=600)


    def plot_graphviz(self,**kwargs):
        fig,ax=plt.subplots(figsize=(7, 6))
        plt.tick_params(direction='out', width=2, length=4, labelsize='large')    #inout
        plt.ylim(0,36+72.5*(self.stage_num-1))
        plt.yticks(np.arange(18,36+72.5*(self.stage_num-1),72.5),self.name_stage())
        plt.ylabel('Developmental stages/Time points',fontsize='large')

        G = nx.DiGraph()
        G.add_nodes_from(self.draw_nodes())  
        G.add_edges_from(self.draw_edges())               #  self.draw_edges()
        pos =graphviz_layout(G, prog='dot')
        size_state=self.adjust_statenum()                       

        labels={}
        for i,nodes in enumerate(G.nodes()):
            labels[nodes]=("".join(nodes).split('_',1))[1]   
        nx.draw_networkx_labels(G, pos, labels, font_size=14)
        nx.draw_networkx(G, ax=ax, alpha=1, edge_color="black", node_color=range(len(size_state)), cmap=plt.cm.YlOrRd, node_size=[i*10 for i in size_state],with_labels=False, pos = pos, arrows=True)     #(原来)font_size=3  
        # [i*2 for i in size_state]     edge_color="silver"
        ax.tick_params(axis='y',left=True,labelleft=True)
        ax.spines['top'].set_color('none')      #隐藏边框
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_color('none')
        plt.show()             #在窗口中显示   plt.show(block=False)
        plt.close()

        output_path=self.__Output_Dir+self.__Output_Name+"/"
        output_name=self.__Output_Name
        fig.savefig(output_path+f"{output_name}_Topology.tif", bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"}, dpi=600)  
        ouput_path=self.__Output_Dir+self.__Output_Name+"/SupplementaryResults/"
        output_name=self.__Output_Name
        np.savetxt(ouput_path+f"{output_name}_adjacency_martrix.txt",np.array(nx.to_numpy_matrix(G)), fmt='%d', delimiter='   ')
                                                            # nx.adjacency_matrix.todense(),,,  nx.to_numpy_matrix(G)


    def run_idti(self):

        # 数据聚类标签的获取
        if True:
            self.cell_clusters()

        # 数据预处理
        if self.__Switch_DeadCellFilter == True:
            self.filter_dead_cell()
        
        if self.__Switch_LowGeneCellsFilter == True:
            self.filter_lowgene_cells()

        if self.__Switch_LowCellGenesFilter == True:
            self.filter_lowcell_genes()
        
        if self.__Switch_Normalize == True :
            self.normalize_data()

        if self.__Switch_LogTransform == True :
            self.log_transform()

        if self.__Switch_HightlyVariableGenes ==  True:  
            self.highly_variable_genes()

        if self.__Switch_ScaleData_MinMax == True :
            self.scaled_data_MinMaxScale()    

        
        if True :
            self.group_stage()
            self.plot_umap()
            self.plot_graphviz()


if __name__=='__main__': 

    adata=sc.read_h5ad("/baimoc/hongyan/work/test/data/1SimulatedData/sim_data.h5ad")                   #输入数据为h5ad文

    # adata=sc.read_h5ad("/baimoc/hongyan/work/test/compare/sim_data/data_source/mouse_sim.h5ad")
    
    cdata = IdtiData(adata)
    cdata.run_idti()