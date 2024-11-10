import csv
import pandas as pd
import matplotlib.pyplot as plt    

def read_csv(file_path):
    data = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return data

name_from_scenario = {
    "1b-1t" : "Single block, single thread",
    "1b-256t": "Single block, 256 threads",
    "MANYb-256t": "Many blocks, 256 threads",
    "q1": "CPU"
}

def plot_results(data, question, filename):
    df = pd.DataFrame(data)
    df = df[df['question'] == question]
    df['K'] = df['K'].astype(int) / 1e6
    df['calculation time'] = df['calculation time'].astype(float)
    
    scenarios = df['scenario'].unique()
    
    for scenario in scenarios:
        scenario_data = df[df['scenario'] == scenario].sort_values(by='K')      
        plt.plot(scenario_data['K'], scenario_data['calculation time'], label=name_from_scenario[scenario])
    
    # Also plot q1 data
    q1_data = pd.DataFrame(data)
    q1_data = q1_data[q1_data['question'] == 'q1']
    q1_data['K'] = q1_data['K'].astype(int)/1e6
    q1_data['calculation time'] = q1_data['calculation time'].astype(float)
    q1_scenario_data = q1_data.sort_values(by='K')
    plt.plot(q1_scenario_data['K'], q1_scenario_data['calculation time'], label=name_from_scenario['q1'], linestyle='--')
    
    plt.xlabel('K (millions of elements)')
    plt.ylabel('Calculation Time (microseconds)')
    plt.title(f'Results for {question}')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(filename)
    plt.close()

# NOTE: requires running Generate-Data.sh, which writes to results.csv
def main():
    file_path = 'results.csv'
    data = read_csv(file_path)

    plot_results(data, 'q3', "q4_with_unified.jpg")
    plot_results(data, 'q2', "q4_without_unified.jpg")

if __name__ == "__main__":
    main()