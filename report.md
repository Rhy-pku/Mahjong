

## 一、练习 9.2.2

**可用表达式分析**
全集
E = { a+b , c−a , b+d , b*d , a−d }

### 基本块的 e_gen / e_kill

**B1**
(1) a=1 (2) b=2
e_gen = ∅
e_kill = { a+b , c−a , b+d , b*d , a−d }

**B2**
(3) c=a+b (4) d=c−a
e_gen = { a+b , c−a }
e_kill = { b+d , b*d , a−d }

**B3**
(5) d=b+d
e_gen = { b+d }
e_kill = { c−a , a−d , b*d }

**B4**
(6) d=a+b (7) e=e+1
e_gen = { a+b }
e_kill = { c−a , b+d , b*d , a−d }

**B5**
(8) b=a+b (9) e=c−a
e_gen = { a+b , c−a }
e_kill = { b+d , b*d , a−d }

**B6**
(10) a=b*d (11) b=a−d
e_gen = { b*d , a−d }
e_kill = { a+b , c−a , b+d }

---

### IN / OUT（迭代到不动点）

**B1**
IN = ∅
OUT = ∅

**B2**
IN = ∅
OUT = { a+b , c−a }

**B3**
IN = { a+b , c−a }
OUT = { b+d }

**B4**
IN = { b+d }
OUT = { a+b }

**B5**
IN = { a+b }
OUT = { a+b , c−a }

**B6**
IN = { a+b , c−a }
OUT = { b*d , a−d }

---

## 二、练习 9.2.3

**活跃变量分析**

### def / use

**B1**
def = { a , b }
use = ∅

**B2**
def = { c , d }
use = { a , b }

**B3**
def = { d }
use = { b , d }

**B4**
def = { d , e }
use = { a , b , e }

**B5**
def = { b , e }
use = { a , c }

**B6**
def = { a , b }
use = { d }

---

### IN / OUT（反向，不动点）

**B6**
OUT = ∅
IN = { d }

**B5**
OUT = { d }
IN = { a , c , d }

**B4**
OUT = { a , c , d }
IN = { a , b , c , e }

**B3**
OUT = { a , b , c , e }
IN = { a , b , c , d , e }

**B2**
OUT = { a , b , c , d , e }
IN = { a , b , e }

**B1**
OUT = { a , b , e }
IN = { e }
