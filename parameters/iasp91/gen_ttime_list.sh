#./taup_table -ph Lg -locsat -header heads/iasp91.Lg > ttimes/iasp91.Lg
./taup_table -ph ttP -locsat -header heads/iasp91.P > ttimes/iasp91.P
./taup_table -ph PcP -locsat -header heads/iasp91.PcP > ttimes/iasp91.PcP
./taup_table -ph Pg -locsat -header heads/iasp91.Pg > ttimes/iasp91.Pg #only matches up to deg=9
./taup_table -ph PKKP -locsat -header heads/iasp91.PKKPbc > ttimes/iasp91.PKKPbc
./taup_table -ph PKIKP -locsat -header heads/iasp91.PKP > ttimes/iasp91.PKP
#./taup_table -ph PKP -locsat -header heads/iasp91.PKPab > ttimes/iasp91.PKPab
./taup_table -ph PKP -locsat -header heads/iasp91.PKPbc > ttimes/iasp91.PKPbc
./taup_table -ph Pn -locsat -header heads/iasp91.Pn > ttimes/iasp91.Pn #only matches up to z=35km
./taup_table -ph pP,pPdiff,pPKP,pPKIKP -locsat -header heads/iasp91.pP > ttimes/iasp91.pP #missing depth 0, and some long-distance - I think the soln here is to fill the gaps with Ps?
#./taup_table -ph Rg -locsat -header heads/iasp91.Rg > ttimes/iasp91.Rg
./taup_table -ph ttS -locsat -header heads/iasp91.S > ttimes/iasp91.S
./taup_table -ph ScP -locsat -header heads/iasp91.ScP > ttimes/iasp91.ScP
./taup_table -ph Sn -locsat -header heads/iasp91.Sn > ttimes/iasp91.Sn # missing long distances, and only matches up until z=35

#./taup_table -ph Lg -locsat -iangle -header heads/iasp91.Lg > iangles/iasp91.Lg
./taup_table -ph ttP -locsat -iangle -header heads/iasp91.P > iangles/iasp91.P
./taup_table -ph PcP -locsat -iangle -header heads/iasp91.PcP > iangles/iasp91.PcP
./taup_table -ph Pg -locsat -iangle -header heads/iasp91.Pg > iangles/iasp91.Pg 
./taup_table -ph PKKP -locsat -iangle -header heads/iasp91.PKKPbc > iangles/iasp91.PKKPbc
./taup_table -ph PKIKP -locsat -iangle -iangle -header heads/iasp91.PKP > iangles/iasp91.PKP
#./taup_table -ph PKP -locsat -iangle -iangle -header heads/iasp91.PKPab > iangles/iasp91_angle.PKPab
./taup_table -ph PKP -locsat -iangle -iangle -header heads/iasp91.PKPbc > iangles/iasp91.PKPbc
./taup_table -ph Pn -locsat -iangle -header heads/iasp91.Pn > iangles/iasp91.Pn 
./taup_table -ph pP,pPdiff,pPKP,pPKIKP -locsat -iangle -header heads/iasp91.pP > iangles/iasp91.pP 
#./taup_table -ph Rg -locsat -iangle -header heads/iasp91.Rg > iasp91_angle.Rg
./taup_table -ph ttS -locsat -iangle -header heads/iasp91.S > iangles/iasp91.S
./taup_table -ph ScP -locsat -iangle -header heads/iasp91.ScP > iangles/iasp91.ScP
./taup_table -ph Sn -locsat -iangle -header heads/iasp91.Sn > iangles/iasp91.Sn 
