import h5py

file_path = '../hdf5_graph_files/test_graph_apo_pocket_binary_15_apo.hdf5'

def print_hdf5_file_structure(hdf5_file_path):
    """
    HDF5ファイルの構造を表示する関数。
    :param hdf5_file_path: HDF5ファイルのパス
    """
    def print_group(group, prefix=''):
        """
        再帰的にグループとデータセットを表示するヘルパー関数。
        :param group: HDF5グループオブジェクト
        :param prefix: 現在の階層を表すプレフィックス
        """
        for key in group.keys():
            item = group[key]
            print(f"{prefix}/{key}: {'Group' if isinstance(item, h5py.Group) else 'Dataset'}")
            if isinstance(item, h5py.Group):
                print_group(item, prefix=f"{prefix}/{key}")

    with h5py.File(hdf5_file_path, 'r') as file:
        print_group(file)

# HDF5ファイルのパスを指定して関数を呼び出します。
print_hdf5_file_structure(file_path)

    
        

