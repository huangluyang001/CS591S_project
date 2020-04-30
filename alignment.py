import json

file = open('/home/zhe/workspace/summary_model_output/sota/cnndm/DCA/dca_m7.jsonlist', 'r')
dcas = []
for line in file:
    data = json.loads(line)
    dcas.append(data['original'])




i = 0
file = open('/home/zhe/workspace/summary_model_output/sota/cnndm/align_dca/fuzzy_align/aligned.jsonlist', 'r')
for line in file:
    data = json.loads(line)
    id_dca = data['id_dca']
    id_our = data['id_our']
    dca = dcas[id_dca]
    with open('/data2/luyang/process-cnn/results/DCAm7_aligned/output/' + str(i) + '.dec', 'w') as f:
        for sent in dcas[id_dca]:
            f.write(sent + '\n')

    with open('/data2/luyang/process-cnn/results/DCA/refs/test/' + str(i) + '.ref', 'w') as f:
        with open('/data2/luyang/process-cnn/finished_files/refs/test/' + id_our, 'r') as g:
            for line in g:
                f.write(line.strip() + '\n')
    i += 1

