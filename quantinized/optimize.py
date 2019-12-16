import os

path = './tf_files'
optimized_path = './tf_files_optimized'
round_path = './tf_files_round_weight'

def get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break
    return files

names = list_files(path)

files = get_files(path)
print(files)

# for name in names:
#     os.system('python -m tensorflow.python.tools.optimize_for_inference \
#   --input=tf_files/' + name + '\
#   --output=tf_files_optimized/optimized' + name + '\
#   --input_names="X_inputs" \
#   --output_names="output"')

# names_optimized = list_files(optimized_path)

# for name in names_optimized:
#     os.system('python -m scripts.quantize_graph \
#   --input=tf_files_optimized/' + name + '\
#   --output=tf_files_round_weight/' + name +'_rounded_graph.pb \
#   --output_node_names=output \
#   --mode=weights_rounded')

names_round = list_files(round_path)

for name in names_round:
    os.system('gzip -c tf_files_round_weight/'+name + '> tf_files_round_weight/' + name + '.gz')
    os.system('gzip -l tf_files_round_weight/'+ name + '.gz')
