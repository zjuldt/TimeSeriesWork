%在线监测--发生异常的数据
    true1=1;true2=size(DATAte,1);     %20151102C201-2区筒壁温度异常
    xte(:,1)=DATAte(true1:true2,90);     %烘前叶丝流量
    xte(:,2)=DATAte(true1:true2,86);     %烘前水分
    xte(:,3)=DATAte(true1:true2,61);     %SIROX阀前蒸汽压力
    xte(:,4)=DATAte(true1:true2,63);     %SIROX后叶丝温度
    xte(:,5)=DATAte(true1:true2,68);     %SIROX蒸汽体积流量
    xte(:,6)=DATAte(true1:true2,72);     %SIROX蒸汽质量流量
    xte(:,7)=DATAte(true1:true2,66);     %SIROX蒸汽薄膜阀开度
    xte(:,8)=DATAte(true1:true2,46);     %KLD排潮负压
    xte(:,9)=DATAte(true1:true2,48);     %KLD排潮风门开度
    xte(:,10)=DATAte(true1:true2,59);    %KLD总蒸汽压力
    xte(:,11)=DATAte(true1:true2,9);     %1区蒸汽压力
    xte(:,12)=DATAte(true1:true2,5);     %1区筒壁温度
    xte(:,13)=DATAte(true1:true2,17);    %2区蒸汽压力
    xte(:,14)=DATAte(true1:true2,13);    %2区筒壁温度
    xte(:,15)=DATAte(true1:true2,3);     %1区冷凝水温度
    xte(:,16)=DATAte(true1:true2,11);    %2区冷凝水温度
    xte(:,17)=DATAte(true1:true2,52);    %KLD热风温度
    xte(:,18)=DATAte(true1:true2,50);    %KLD热风风速
    xte(:,19)=DATAte(true1:true2,38);    %KLD除水量
    xte(:,20)=DATAte(true1:true2,40);    %KLD烘后水分
    xte(:,21)=DATAte(true1:true2,42);    %KLD烘后温度
    xte(:,22)=DATAte(true1:true2,92);    %冷却水分
    xte(:,23)=DATAte(true1:true2,94);    %冷却温度