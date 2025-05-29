import json
from scipy.stats import pearsonr, spearmanr

def read_scores(file_path):
    labels = []
    predictions = []
    
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                label_score = float(data['label'].split('score of this image is ')[1].strip('.'))
                # import ipdb;ipdb.set_trace()
                if '<' in data['predict'] and '>' in data['predict']:
                    start =  data['predict'].index('<')
                    end =  data['predict'].index('>')
                    predict_score = float(data['predict'][start+1: end])
                else:
                    predict_score = float(data['predict'].split('score of this image is ')[1].strip('.'))
                
                labels.append(label_score)
                predictions.append(predict_score)
            except:
                print('失败')
                # import ipdb;ipdb.set_trace()
    
    return labels, predictions

# 计算 PLCC 和 SRCC
def calculate_correlations(labels, predictions):
    plcc, _ = pearsonr(labels, predictions)
    srcc, _ = spearmanr(labels, predictions)
    return plcc, srcc

def main(file_path):
    labels, predictions = read_scores(file_path)
    plcc, srcc = calculate_correlations(labels, predictions)
    print(f"PLCC: {plcc}")
    print(f"SRCC: {srcc}")

if __name__ == '__main__':
    import sys
    result_path = sys.argv[1]
    main(result_path)

