Lstr=1;
% DATAi{1}=datan(Lstr:end,:);
DATAi{1}=xte(300:600,:);
conflevel=0.99;
[datain,meand,stdd]=autoscale(DATAi{1});
[P0 ,L0,G0]=pcacov(datain'*datain/(size(datain,1)-1));
[T02,mean_Tnn ,invcov_Tnn ,ctr_Tnn ]=model_development_vtsingular1(datain,P0,size(P0,2),conflevel);
xf=xte(676:706,:);%fault 1: 676-744 is fault duration
% xf=xte(500:543,:);%fault 2
% xf=xn2(300:600,:);
DATAi{2}=xf;
xten=autoscale(DATAi{2},meand,stdd);
[Tf2te,speftemp]=online_monitoring(xten,P0,P0,mean_Tnn ,invcov_Tnn );
%[Tf2te,speftemp]=online_monitoring(datain,P0,P0,mean_Tnn ,invcov_Tnn );

figure
plot(Tf2te,'k.-')
hold on
plot(ctr_Tnn*ones(1,size(Tf2te,1)),'r:')
xlabel('Samples')
ylabel('D^2')


close all
for j=1:23
    figure
    plot(DATAi{1}(:,j))
    hold on
plot(DATAi{2}(:,j),'r')
end




conflevel=0.99;
numalarm=30;
[varseln,varself,DATA0n,DATA2n,mostfretfj,SRVCtfj,RVCtfjbp]=fault_variable_selection_NeLFDA(DATAi,5,conflevel,numalarm);
[DATA2nall,indexfaultt]=addback_matrix(DATA2n,varself,mostfretfj);

[varseln,varself,DATA0n,DATA2n,mostfretfj,SRVCtfj,RVCtfjbp]=fault_variable_selection_NeLFDA_check(DATAi,5,conflevel,numalarm,3);
[DATA2nall,indexfaulttcheck]=addback_matrix(DATA2n,varself,mostfretfj);

[DATA2tesele,DATA2teleft]=select_matrix(DATAi{2},indexfaultt);
[DATA0tesele,DATA0teleft]=select_matrix(DATAi{1},indexfaultt);
DATAnii{1}=DATA0teleft;DATAnii{2}=DATA2teleft;DATAsum=[DATA0teleft;DATA2teleft];
[P0 ,L0,G0]=pcacov(DATAsum'*DATAsum/(size(DATAsum,1)-1));
conflevel=0.99;
[T02,mean_Tn ,invcov_Tn ,ctr_Tn ]=model_development_vtsingular1(DATAsum,P0 ,size(P0 ,2),conflevel);
[Tf2,speftemp]=online_monitoring(DATA2teleft,P0 ,P0 ,mean_Tn ,invcov_Tn );
[Tf0,speftemp]=online_monitoring(DATA0teleft,P0 ,P0 ,mean_Tn ,invcov_Tn );
figure
subplot(211)
plot(Tf0,'k.-')
hold on
plot(ctr_Tn *ones(1,size(Tf0,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for general variables in normal data')
subplot(212)
plot(Tf2,'k.-')
hold on
plot(ctr_Tn *ones(1,size(Tf2,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for general variables in fault data')

DATAfii{1}=DATA0tesele;DATAfii{2}=DATA2tesele;FDAnum=rank([DATA0tesele;DATA2tesele]);
[Rsingfi ]=NeLFDA(DATAfii,FDAnum);
conflevel=0.99;
[T02,mean_Tnn ,invcov_Tnn ,ctr_Tnn ]=model_development_vtsingular1(DATAfii{1},Rsingfi {1},size(Rsingfi {1},2),conflevel);
[Tf2,mean_Tf ,invcov_Tf ,ctr_Tf ]=model_development_vtsingular1(DATAfii{2},Rsingfi {2},size(Rsingfi {2},2),conflevel);

figure
subplot(211)
plot(T02,'k.-')
hold on
plot(ctr_Tnn *ones(1,size(T02,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for fauty variables in normal data')
subplot(212)
plot(Tf2,'k.-')
hold on
plot(ctr_Tf *ones(1,size(Tf2,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for fauty variables in fault data')








Lstr=1;
DATAite{1}=datante(Lstr:end,:);
DATAite{1}=DATAi{1};
magB=10;
Nt=size(DATAite{1},1);
x1n1te=magB*ones(Nt,1);
magB=10;
tao=100;
exponnoise= -magB*tao*exppdf(0:1000,tao)+magB;
x1n2te=(exponnoise(1,1:Nt))';



DATAite{2}=DATAite{1};
Nte=size(DATAite{2},1);
DATAite{2}(:,11)=DATAite{1}(:,11)+x1n1te(1:size(DATAite{2},1))*0.8+randn(Nte,1)*stdvar(1);
DATAite{2}(:,12)=DATAite{2}(:,12)+x1n2te(1:size(DATAite{2},1))*0.3+randn(Nte,1)*stdvar(2);
DATAite{2}(:,15)=DATAite{1}(:,15)+x1n2te(1:size(DATAite{2},1))*0.3+randn(Nte,1)*stdvar(3);
DATAite{2}(:,20)=DATAite{2}(:,20)-x1n2te(1:size(DATAite{2},1))*0.4-randn(Nte,1)*stdvar(4);
DATAite{2}(:,21)=DATAite{1}(:,21)+x1n2te(1:size(DATAite{2},1))*0.5+randn(Nte,1)*stdvar(5);
DATAite{2}(:,22)=DATAite{2}(:,22)-x1n2te(1:size(DATAite{2},1))*0.3-randn(Nte,1)*stdvar(6);
DATAite{2}(:,23)=DATAite{2}(:,23)+x1n2te(1:size(DATAite{2},1))*0.45+randn(Nte,1)*stdvar(7);
for j=1:7
figure
plot(DATAite{2}(1:100,varindex(j)),'r.:')
hold on
plot(DATAite{1}(1:100,varindex(j)),'k.:')
end
close all
Ltr=100;
DATA0te=DATAite{1};
DATA2te=DATAite{2}(1:Ltr,:);
% DATA0te=DATAi{1};
% DATA2te=DATAi{2}(1:Ltr,:);

[DATA2tesele,DATA2teleft]=select_matrix(DATA2te,indexfaultt);
[DATA0tesele,DATA0teleft]=select_matrix(DATA0te,indexfaultt);
[Tf2te,speftemp]=online_monitoring(DATA2teleft,P0 ,P0 ,mean_Tn ,invcov_Tn );
[Tf0te,speftemp]=online_monitoring(DATA0teleft,P0 ,P0 ,mean_Tn ,invcov_Tn );
figure
subplot(211)
plot(Tf0te,'k.-')
hold on
plot(ctr_Tn *ones(1,size(Tf0te,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for general variables in normal data')
subplot(212)
plot(Tf2te,'k.-')
hold on
plot(ctr_Tn *ones(1,size(Tf2te,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for general variables in fault data')

[Tf2te,speftemp]=online_monitoring(DATA2tesele,Rsingfi{2},Rsingfi{2},mean_Tf ,invcov_Tf );
[Tf0te,speftemp]=online_monitoring(DATA0tesele,Rsingfi{1},Rsingfi{1},mean_Tnn ,invcov_Tnn );

figure
subplot(211)
plot(Tf0te,'k.-')
hold on
plot(ctr_Tnn *ones(1,size(Tf0te,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for fauty variables in normal data')
subplot(212)
plot(Tf2te,'k.-')
hold on
plot(ctr_Tf *ones(1,size(Tf2te,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for fauty variables in fault data')


[Tf2te,speftemp]=online_monitoring(DATA2teleft,P0 ,P0 ,mean_Tn ,invcov_Tn );
figure
subplot(211)
plot(Tf2te,'k.-')
hold on
plot(ctr_Tn*ones(1,size(Tf2te,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for general variables')
[Tf2te,speftemp]=online_monitoring(DATA2tesele,Rsingfi{2},Rsingfi{2},mean_Tf ,invcov_Tf );
subplot(212)
plot(Tf2te,'k.-')
hold on
plot(ctr_Tf*ones(1,size(Tf2te,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for fauty variables')
