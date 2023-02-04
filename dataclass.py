import copy

class national_illness:
    def __init__(self,):
        super().__init__()
        self.item_list = ['OT']
        # self.item_list = ['%WEIGHTED ILI',
        #                   '%UNWEIGHTED ILI',
        #                   'AGE 0-4',
        #                   'AGE 5-24',
        #                   'ILITOTAL',
        #                   'NUM. OF PROVIDERS',
        #                   'OT']
        indicators = {'mse': 100, 'mae': 100,'mape':100}
        items_dic = {'%WEIGHTED ILI': copy.deepcopy(indicators),
                     '%UNWEIGHTED ILI': copy.deepcopy(indicators),
                     'AGE 0-4': copy.deepcopy(indicators),
                     'AGE 5-24': copy.deepcopy(indicators),
                     'ILITOTAL': copy.deepcopy(indicators),
                     'NUM. OF PROVIDERS': copy.deepcopy(indicators),
                     'OT': copy.deepcopy(indicators)}
        self.reuslts_dic = {60: copy.deepcopy(items_dic),
                            48: copy.deepcopy(items_dic),
                            36: copy.deepcopy(items_dic),
                            24: copy.deepcopy(items_dic)}
        self.finalreuslts_dic = {60: copy.deepcopy(indicators),
                                 48: copy.deepcopy(indicators),
                                 36: copy.deepcopy(indicators),
                                 24: copy.deepcopy(indicators),}


class weather:
    def __init__(self,):
        super().__init__()
        self.item_list = ['OT']
        # self.item_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 'OT']
        indicators = {'mse': 100, 'mae': 100, 'mape': 100}
        items_dic = {'0': copy.deepcopy(indicators),
                      '1': copy.deepcopy(indicators),
                      '2': copy.deepcopy(indicators),
                      '3': copy.deepcopy(indicators),
                      '4': copy.deepcopy(indicators),
                      '5': copy.deepcopy(indicators),
                      '6': copy.deepcopy(indicators),
                      '7': copy.deepcopy(indicators),
                      '8': copy.deepcopy(indicators),
                      '9': copy.deepcopy(indicators),
                      '10': copy.deepcopy(indicators),
                      '11': copy.deepcopy(indicators),
                      '12': copy.deepcopy(indicators),
                      '13': copy.deepcopy(indicators),
                     'OT': copy.deepcopy(indicators)}
        self.reuslts_dic = {720: copy.deepcopy(items_dic),
                            336: copy.deepcopy(items_dic),
                            192: copy.deepcopy(items_dic),
                            96: copy.deepcopy(items_dic)}
        self.finalreuslts_dic = {720: copy.deepcopy(indicators),
                                 336: copy.deepcopy(indicators),
                                 192: copy.deepcopy(indicators),
                                 96: copy.deepcopy(indicators)}


class traffic:
    def __init__(self,):
        super().__init__()
        # self.item_list = ['0',
        #                   '1',
        #                   '2',
        #                   '3',
        #                   '4',
        #                   '5',
        #                   'OT']
        self.item_list = ['OT']
        indicators = {'mse': 100, 'mae': 100,'mape':100}
        items_dic = {'0': copy.deepcopy(indicators),
                     '1': copy.deepcopy(indicators),
                     '2': copy.deepcopy(indicators),
                     '3': copy.deepcopy(indicators),
                     '4': copy.deepcopy(indicators),
                     '5': copy.deepcopy(indicators),
                     'OT': copy.deepcopy(indicators)}
        self.reuslts_dic = {720: copy.deepcopy(items_dic),
                            336: copy.deepcopy(items_dic),
                            192: copy.deepcopy(items_dic),
                            96: copy.deepcopy(items_dic)}
        self.finalreuslts_dic = {720: copy.deepcopy(indicators),
                                 336: copy.deepcopy(indicators),
                                 192: copy.deepcopy(indicators),
                                 96: copy.deepcopy(indicators)}


class exchange_rate:
    def __init__(self,):
        super().__init__()
        self.item_list = ['OT']
        # self.item_list = ['0', '1', '2', '3', '4', '5', '6', 'OT']
        indicators = {'mse': 100, 'mae': 100,'mape':100}
        items_dic = {'0': copy.deepcopy(indicators),
                     '1': copy.deepcopy(indicators),
                     '2': copy.deepcopy(indicators),
                     '3': copy.deepcopy(indicators),
                     '4': copy.deepcopy(indicators),
                     '5': copy.deepcopy(indicators),
                     '6': copy.deepcopy(indicators),
                     'OT': copy.deepcopy(indicators)}
        self.reuslts_dic = {720: copy.deepcopy(items_dic),
                            336: copy.deepcopy(items_dic),
                            192: copy.deepcopy(items_dic),
                            96: copy.deepcopy(items_dic)}
        self.finalreuslts_dic = {720: copy.deepcopy(indicators),
                                 336: copy.deepcopy(indicators),
                                 192: copy.deepcopy(indicators),
                                 96: copy.deepcopy(indicators)}


class ETT:
    def __init__(self,):
        super().__init__()
        # self.item_list = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        self.item_list = ['OT']
        indicators = {'mse': 100, 'mae': 100,'mape':100}
        items_dic = {'HUFL': copy.deepcopy(indicators),
                     'HULL': copy.deepcopy(indicators),
                     'MUFL': copy.deepcopy(indicators),
                     'MULL': copy.deepcopy(indicators),
                     'LUFL': copy.deepcopy(indicators),
                     'LULL': copy.deepcopy(indicators),
                     'OT': copy.deepcopy(indicators)}

        self.reuslts_dic = {720: copy.deepcopy(items_dic),
                            336: copy.deepcopy(items_dic),
                            192: copy.deepcopy(items_dic),
                            96: copy.deepcopy(items_dic)}
        self.finalreuslts_dic = {720: copy.deepcopy(indicators),
                                 336: copy.deepcopy(indicators),
                                 192: copy.deepcopy(indicators),
                                 96: copy.deepcopy(indicators)}


class electricity:
    def __init__(self,):
        super().__init__()
        self.item_list = ['OT']
        # self.item_list = ['0', '1', '2', '3', '4', '5', '6', 'OT']
        indicators = {'mse': 100, 'mae': 100,'mape':100}
        items_dic = {'0': copy.deepcopy(indicators),
                     '1': copy.deepcopy(indicators),
                     '2': copy.deepcopy(indicators),
                     '3': copy.deepcopy(indicators),
                     '4': copy.deepcopy(indicators),
                     '5': copy.deepcopy(indicators),
                     '6': copy.deepcopy(indicators),
                     'OT': copy.deepcopy(indicators)}

        self.reuslts_dic = {720: copy.deepcopy(items_dic),
                            336: copy.deepcopy(items_dic),
                            192: copy.deepcopy(items_dic),
                            96: copy.deepcopy(items_dic)}
        self.finalreuslts_dic = {720: copy.deepcopy(indicators),
                                 336: copy.deepcopy(indicators),
                                 192: copy.deepcopy(indicators),
                                 96: copy.deepcopy(indicators)}


class wind:
    def __init__(self,):
        super().__init__()
        self.item_list = ['OT']
        # self.item_list = ['0', '1', '2', '3', '4', '5', '6', 'OT']
        indicators = {'mse': 100, 'mae': 100,'mape':100}
        items_dic = {'OT': copy.deepcopy(indicators)}

        self.reuslts_dic = {10: copy.deepcopy(items_dic)}
        self.finalreuslts_dic = {10: copy.deepcopy(indicators)}