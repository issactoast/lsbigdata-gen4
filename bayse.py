import numpy as np
p=np.array([0.16, 0.18, 0.20]) / 0.54
bug_p=np.array([0.05, 0.02, 0.03])

# P(2022|B)=?
# P(2022 교 B) / P(B)
# P(B)=?
sum(bug_p*p)
# P(2022 교 B)=?
# P(2022)*P(B|2022) = 
p[0]*bug_p[0]
p[0]*bug_p[0] / sum(bug_p*p)

### 수니, 뭉이, 젤리 문제
import numpy as np
p=np.array([0.5, 0.3, 0.2]) # 사전분포
break_p=np.array([0.01, 0.02, 0.03])

# P(S|B):접시가 깨졌을때 수니가 일하고 있는 확률
p[0]*break_p[0] / sum(break_p*p)
# P(J|B):접시가 깨졌을때 젤리가 일하고 있는 확률
p[1]*break_p[1] / sum(break_p*p)
# P(M|B):접시가 깨졌을때 뭉이가 일하고 있는 확률
p[2]*break_p[2] / sum(break_p*p)

# 사후분포 (접시 한번 깨짐)
# P(S|B) = 0.294
# P(J|B) = 0.353
# P(M|B) = 0.353
p=np.array([0.294, 0.353, 0.353]) # 사전분포
p[0]*break_p[0] / sum(break_p*p)
p[1]*break_p[1] / sum(break_p*p)
p[2]*break_p[2] / sum(break_p*p)

# 사후분포 (접시 두번 깨짐)
# P(S|B) = 0.143
# P(J|B) = 0.343
# P(M|B) = 0.514

