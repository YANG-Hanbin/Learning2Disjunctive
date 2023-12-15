import matplotlib.pyplot as plt

fn='tr12-30'
fn='beasleyC3'
fn='assign1-5-8'
f=open(fn,'r')
mode=0
rec=[[],[],[]]
for line in f:
    line=line.replace('\r','').replace('\n','')
    if 'MFR:' in line:
        line=line.split(':')[-1].split(' ')[-1][:-1]
        rec[0].append(float(line))
        mode=1
    elif 'Vanilla:' in line:
        line=line.split(':')[-1].split(' ')[-1][:-1]
        rec[1].append(float(line))
        mode=2
    elif 'RL:' in line:
        line=line.split(':')[-1].split(' ')[-1][:-1]
        rec[2].append(float(line))
        mode=3
    elif mode==1:
        if '---' in line or '#' in line:
            continue
        line=line.replace(' ','').split('|')
        line=[x for x in line if x!=''][2]
        rec[0].append(float(line))

    elif mode==2:
        if '---' in line or '#' in line:
            continue
        line=line.replace(' ','').split('|')
        line=[x for x in line if x!=''][2]
        rec[1].append(float(line))
    elif mode==3:
        if '---' in line or '#' in line:
            continue
        line=line.replace(' ','').split('|')
        line=[x for x in line if x!=''][2]
        rec[2].append(float(line))
f.close()

print(rec)

x=[]
for i in range(len(rec[0])):
    x.append(i)
    
plt.plot(x,rec[0],label='MFR')
plt.plot(x,rec[2],label='RL')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Best Bound')
plt.savefig(f'../fig/mfrRL{fn}.png')
plt.clf()
    
plt.plot(x,rec[1],label='Vanilla')
plt.plot(x,rec[2],label='RL')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Best Bound')
plt.savefig(f'../fig/vanRL{fn}.png')
plt.clf()