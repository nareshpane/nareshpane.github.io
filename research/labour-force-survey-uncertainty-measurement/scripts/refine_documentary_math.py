"""Source-level line breaks based on expanded 320/390px MathJax measurements."""
from pathlib import Path
import re
P=Path(__file__).resolve().parents[2]/'labour-force-survey-uncertainty-measurement.html'
h=P.read_text(encoding='utf-8')

replacements={
 r'\pi_{ij}=P_d(\text{cluster }i\text{ selected})\,P_d(\text{dwelling }j\text{ selected}\mid i)':r'\begin{aligned}\pi_{ij}&=P_d(\text{cluster }i\text{ selected})\\&\quad\times P_d(\text{dwelling }j\text{ selected}\mid i)\end{aligned}',
 r'\pi_{hij}=P(i\in s_h)P(j\in s_{hi}\mid i\in s_h)=\pi_{hi}\pi_{hj\mid i}':r'\begin{aligned}\pi_{hij}&=P(i\in s_h)\\&\quad\times P(j\in s_{hi}\mid i\in s_h)\\&=\pi_{hi}\pi_{hj\mid i}\end{aligned}',
 r'V_d(\hat Y_t-\hat Y_{t-1})=V_d(\hat Y_t)+V_d(\hat Y_{t-1})-2\operatorname{Cov}_d(\hat Y_t,\hat Y_{t-1}).':r'\begin{aligned}V_d(\hat Y_t-\hat Y_{t-1})&=V_d(\hat Y_t)+V_d(\hat Y_{t-1})\\&\quad-2\operatorname{Cov}_d(\hat Y_t,\hat Y_{t-1}).\end{aligned}',
 r'E_d[(A-B)^2]=E_d[A^2]+E_d[B^2]-2E_d[AB]':r'\begin{aligned}E_d[(A-B)^2]&=E_d[A^2]+E_d[B^2]\\&\quad-2E_d[AB]\end{aligned}',
 r'SE_\Delta=\sqrt{SE_t^2+SE_{t-1}^2-2\rho SE_tSE_{t-1}}':r'\begin{gathered}v_\Delta=SE_t^2+SE_{t-1}^2\\\phantom{v_\Delta={}}-2\rho SE_tSE_{t-1},\\SE_\Delta=\sqrt{v_\Delta}.\end{gathered}',
 r'\tilde y_i=\begin{cases}\text{employed, absent},&\text{job}_i=1,\ H_i=0,\ A_i=\text{vacation},\\\text{unresolved},&\text{otherwise}.\end{cases}':r'\begin{gathered}B_i=\{\text{job}_i=1,\ H_i=0,\\ A_i=\text{vacation}\},\\[6pt]\tilde y_i=\begin{cases}\text{employed, absent},&B_i\text{ holds},\\\text{unresolved},&\text{otherwise}.\end{cases}\end{gathered}',
 r'\text{when the prior value is valid and applicable}.':r'\text{valid, applicable prior value}.',
 r'\mathcal D_i=\{k:R_k=1,\ C(x_k)=C(x_i),\ k\ne i\}':r'\begin{gathered}\mathcal D_i=\{k:R_k=1,\ k\ne i,\\ C(x_k)=C(x_i)\}\end{gathered}',
 r'\hat T_{imp}-\hat T_{complete}=\sum_i w_i(1-R_i)(\tilde y_i-y_i)':r'\begin{gathered}\hat T_{imp}-\hat T_{complete}\\=\sum_i w_i(1-R_i)(\tilde y_i-y_i)\end{gathered}',
 r'V(\hat T_{imp})=V_{s,R}\{E_D(\hat T_{imp}\mid s,R)\}+E_{s,R}\{V_D(\hat T_{imp}\mid s,R)\}.':r'\begin{aligned}V(\hat T_{imp})&=V_{s,R}\{E_D(\hat T_{imp}\mid s,R)\}\\&\quad+E_{s,R}\{V_D(\hat T_{imp}\mid s,R)\}.\end{aligned}',
 r'P(R\mid X,Y_{obs},Y_{mis})=P(R\mid X,Y_{obs})':r'\begin{gathered}P(R\mid X,Y_{obs},Y_{mis})\\=P(R\mid X,Y_{obs})\end{gathered}',
 r'\hat R-R\approx\frac{(\hat T_y-T_y)-R(\hat T_x-T_x)}{T_x}.':r'\begin{aligned}\hat R-R&\approx\frac{\hat T_y-T_y}{T_x}\\&\quad-R\frac{\hat T_x-T_x}{T_x}.\end{aligned}',
 r'V_d(\hat R)\approx T_x^{-2}[V_d(\hat T_y)+R^2V_d(\hat T_x)-2R\operatorname{Cov}_d(\hat T_y,\hat T_x)]':r'\begin{aligned}V_d(\hat R)&\approx T_x^{-2}\bigl[\\&\quad V_d(\hat T_y)+R^2V_d(\hat T_x)\\&\quad-2R\operatorname{Cov}_d(\hat T_y,\hat T_x)\bigr]\end{aligned}',
 r'\operatorname{MSE}_d(\hat\theta)=E_d[(\hat\theta-\theta)^2]=V_d(\hat\theta)+\{E_d\hat\theta-\theta\}^2.':r'\begin{aligned}\operatorname{MSE}_d(\hat\theta)&=E_d[(\hat\theta-\theta)^2]\\&=V_d(\hat\theta)\\&\quad+\{E_d\hat\theta-\theta\}^2.\end{aligned}',
 r'V_d(\hat T)=V_1\{E_2(\hat T\mid s_1)\}+E_1\{V_2(\hat T\mid s_1)\}.':r'\begin{aligned}V_d(\hat T)&=V_1\{E_2(\hat T\mid s_1)\}\\&\quad+E_1\{V_2(\hat T\mid s_1)\}.\end{aligned}',
 r'V_*\!\left(\sum_i\frac{n_h}{n_h-1}m_{hi}a_{hi}\right)=\frac{n_h}{n_h-1}\sum_i(a_{hi}-\bar a_h)^2.':r'\begin{gathered}V_*\!\left(\sum_i\frac{n_h}{n_h-1}m_{hi}a_{hi}\right)\\=\frac{n_h}{n_h-1}\sum_i(a_{hi}-\bar a_h)^2.\end{gathered}',
 r'30(18)+45(24)+25(20)+60(16)=3,080':r'\begin{aligned}&30(18)+45(24)\\&\quad+25(20)+60(16)=3,080\end{aligned}',
 r'0.05\sqrt{0.04/0.96}\sqrt{0.955/0.045}=4.702\%':r'\begin{gathered}0.05\sqrt{\frac{0.04}{0.96}}\sqrt{\frac{0.955}{0.045}}\\=4.702\%\end{gathered}',
 r'0.05\sqrt{0.04/0.96}\sqrt{0.955/0.045}=0.0470185':r'\begin{gathered}0.05\sqrt{\frac{0.04}{0.96}}\sqrt{\frac{0.955}{0.045}}\\=0.0470185\end{gathered}',
 r'\widehat V(\bar y)=(1-400/10,000)0.0736/400=0.00017664':r'\begin{aligned}\widehat V(\bar y)&=\left(1-\frac{400}{10,000}\right)\frac{0.0736}{400}\\&=0.00017664\end{aligned}',
 r'[400+0.2^2(2,500)-2(0.2)(500)]/1,000^2=0.0003':r'\begin{gathered}\frac{\begin{gathered}400+0.2^2(2,500)\\-2(0.2)(500)\end{gathered}}{1,000^2}\\=0.0003\end{gathered}',
 r'1,000^2(0.9)0.09/100+2,000^2(0.9)0.16/200=810+2,880=3,690':r'\begin{aligned}\widehat V&=\frac{1,000^2(0.9)0.09}{100}\\&\quad+\frac{2,000^2(0.9)0.16}{200}\\&=810+2,880\\&=3,690\end{aligned}',
}
for old,new in replacements.items():
    if old not in h:
        print('Already revised or absent:',old[:65]);continue
    # Replacements needing multiple lines cannot remain inline.
    inline=r'\('+old+r'\)'
    if inline in h:
        h=h.replace(inline,'</p><div class="equation">\\['+new+'\\]</div><p>')
    else:h=h.replace(old,new)
# Move all substantive RHC expressions out of prose, even when TeX source is short.
start=h.index('<details open><summary>Rao');end=h.index('</details>',start)
section=h[start:end]
section=re.sub(r'\\\(([^\n]*?)\\\)',lambda m:'</p><div class="equation">\\['+m[1]+'\\]</div><p>' if len(m[1])>30 else m[0],section)
section=section.replace(r'\sum_g\{M_g\sum_{i\in g}T_i^2/M_i-T_g^2\}',r'\begin{gathered}V(\hat T_h\mid G)\\=\sum_g\left\{M_g\sum_{i\in g}\frac{T_i^2}{M_i}-T_g^2\right\}\end{gathered}')
h=h[:start]+section+h[end:]
# Preserve prose punctuation next to a promoted display without empty paragraphs.
h=re.sub(r'</div><p>\s*([.,;])\s*',r'</div><p>',h)
h=re.sub(r'<p>\s*</p>','',h)
h=h.replace('the seven-stage visual is a teaching sequence','the animated visual is a teaching sequence')
h=h.replace('Different building types illustrate a dwelling frame. Highlighted dwellings represent a small conceptual sample; visible household attributes do not determine selection and the numbers are not the LFS sampling fraction.', 'Different building types locate illustrative dwellings in a heterogeneous frame. An amber window marks one selected dwelling, not an entire selected building. Visible household attributes do not determine selection and these numbers are not the LFS sampling fraction.')
# More legible covariance identity in the provincial ratio discussion.
def provincial(m):
    tex=m[1]
    if 'CV' in tex and 'rho' in tex and len(tex)>40:
        tex=tex.replace('-2',r'\\&\quad-2',1)
        tex=r'\begin{aligned}'+tex.replace(r'\approx',r'&\approx',1)+r'\end{aligned}'
        return '</p><div class="equation">\\['+tex+'\\]</div><p>'
    return m[0]
start=h.index('id="provinces"');end=h.index('id="case-study"',start)
h=h[:start]+re.sub(r'\\\((.*?)\\\)',provincial,h[start:end])+h[end:]
P.write_text(h,encoding='utf-8')
print('Reformatted measured long equations; equivalent formulas and numerical values retained.')
app=P.parent/'labour-force-survey-uncertainty-measurement/js/app.js'
js=app.read_text(encoding='utf-8').replace('Pause sampled people','Pause selection').replace('Animate sampled people','Animate selected dwellings')
app.write_text(js,encoding='utf-8')
