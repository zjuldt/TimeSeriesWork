Lstr=1;
DATAi{1}=datan(Lstr:end,:);
for i=1:size(DATAi{1},1)
    DATAitemp=DATAi{1};
    DATAitemp(i,:)=[];
    meani(i,:)=mean(DATAitemp);
end
J=size(meani,2);
for j=1:size(meani,2)
figure
plot(meani(:,j))
end
close all
numM=200;numS=ceil(2*size(DATAi{1},1)/3);
[meani,junzhi,biaozhuncha]=NeLFDA_checkmontecarlo(DATAi{1},numM,numS);

% junzhi=mean(meani);
% biaozhuncha=std(meani);
varindex=[11 12 15 20 21 22 23];
for j=1:7
   stdvar(1,j)=std(DATAi{1}(:,varindex(j)));
end

magB=10;
Nt=size(DATAi{1},1);
x1n1=magB*ones(Nt,1);
magB=10;
tao=100;
exponnoise= -magB*tao*exppdf(0:1000,tao)+magB;
x1n2=(exponnoise(1,1:Nt))';
DATAi{2}=DATAi{1};
DATAi{2}(:,11)=DATAi{1}(:,11)+x1n1*0.8+randn(Nt,1)*stdvar(1);
DATAi{2}(:,12)=DATAi{2}(:,12)+x1n2*0.3+randn(Nt,1)*stdvar(2);
DATAi{2}(:,15)=DATAi{1}(:,15)+x1n2*0.3+randn(Nt,1)*stdvar(3);
DATAi{2}(:,20)=DATAi{2}(:,20)-x1n2*0.4-randn(Nt,1)*stdvar(4);
DATAi{2}(:,21)=DATAi{1}(:,21)+x1n2*0.5+randn(Nt,1)*stdvar(5);
DATAi{2}(:,22)=DATAi{2}(:,22)-x1n2*0.3-randn(Nt,1)*stdvar(6);
DATAi{2}(:,23)=DATAi{2}(:,23)+x1n2*0.45+randn(Nt,1)*stdvar(7);
close all
for j=1:7
figure
plot(DATAi{2}(1:50,varindex(j)),'r.:')
hold on
plot(DATAi{1}(1:50,varindex(j)),'k.:')
end


Ltr=50;
DATAi{1}=DATAi{1};
%DATAi{2}=[DATAi{1}(76:100,:);DATAi{2}(1:25,:)];
DATAi{2}=DATAi{2}(1:50,:);
% [DATAfiin1{1},meandf1,stddf1]=autoscale(DATAi{1});
% DATAfiin1{2}=autoscale(DATAi{2},meandf1,stddf1);
% [Rsingfi1]=NeLFDA(DATAfiin1,2);
% Tnsele1=DATAfiin1{1}*Rsingfi1{1};
% Tfsele1=DATAfiin1{2}*Rsingfi1{2};
% figure
% plot(Tnsele1(:,1),Tnsele1(:,2),'k*')
% hold on
% plot(Tfsele1(:,1),Tfsele1(:,2),'ro')
% xlabel('the first component')
% ylabel('the second component')
% legend('normal data','fault data')



conflevel=0.99;
numalarm=30;alpha=3;
[varseln,varself,DATA0n,DATA2n,mostfretfj,SRVCtfj,RVCtfjbp]=fault_variable_selection_NeLFDA_check(DATAi,5,conflevel,numalarm,alpha,numM,numS);
[DATA2nall,indexfaultt]=addback_matrix(DATA2n,varself,mostfretfj);
[DATA2tesele,DATA2teleft]=select_matrix(DATAi{2},indexfaultt);
[DATA0tesele,DATA0teleft]=select_matrix(DATAi{1},indexfaultt);
DATAnii{1}=DATA0teleft;DATAnii{2}=DATA2teleft;DATAsum=[DATA0teleft;DATA2teleft];
[datain,meand,stdd]=autoscale(DATAsum);
[P0 ,L0,G0]=pcacov(datain'*datain/(size(datain,1)-1));
conflevel=0.99;
[T02g,mean_Tn ,invcov_Tn ,ctr_Tn ]=model_development_vtsingular1(datain,P0 ,size(P0,2),conflevel);
DATA2teleftn=autoscale(DATA2teleft,meand,stdd);
DATA0teleftn=autoscale(DATA0teleft,meand,stdd);
[Tf2g,speftemp]=online_monitoring(DATA2teleftn,P0,P0 ,mean_Tn ,invcov_Tn );
[Tf0g,speftemp]=online_monitoring(DATA0teleftn,P0,P0 ,mean_Tn ,invcov_Tn );
figure
subplot(211)
plot(Tf0g,'k.-')
hold on
plot(ctr_Tn *ones(1,size(Tf0g,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for general variables in normal data')
subplot(212)
plot(Tf2g,'k.-')
hold on
plot(ctr_Tn *ones(1,size(Tf2g,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for general variables in fault data')
fg = ksdensity(Tf0g,Tf2g);
figure
plot(fg,'k.:')
xlabel('samples')
ylabel('probability')
% figure
% for i=1:4
% subplot(2,2,i)
% bar(RVCtfjbp{i})
% xlabel('Variable #')
% ylabel('Ratio of contribution')
% % title(['For Simulated Fault #',num2str(ii)])
% xlim([0 23])
% end



DATAfii{1}=DATA0tesele;DATAfii{2}=DATA2tesele;
[mu,junzhif,biaozhunchaf]=NeLFDA_checkmontecarlo(DATAfii{1},numM,numS);
[DATAfiin{1},meandf,stddf]=autoscale(DATAfii{1});
%DATAfii{2}(1:5,:)=DATAfii{1}(300:304,:);
DATAfiin{2}=autoscale(DATAfii{2},meandf,stddf);
[P0f,L0f,G0f]=pcacov(DATAfiin{1}'*DATAfiin{1}/(size(DATAfiin{1},1)-1));
FDAnum=rank([DATAfiin{1};DATAfiin{2}]);
[Rsingfi]=NeLFDA(DATAfiin,FDAnum);
[T02f,mean_Tnn,invcov_Tnn,ctr_Tnn]=model_development_vtsingular1(DATAfiin{1},P0f,size(P0f,2),conflevel);
[Tf2f,speftemp]=online_monitoring(DATAfiin{2},P0f,P0f,mean_Tnn ,invcov_Tnn );
ff= ksdensity(T02f,Tf2f);
figure
plot(ff,'k.:')
xlabel('samples')
ylabel('probability')

figure
subplot(211)
plot(T02f,'k.-')
hold on
plot(ctr_Tnn *ones(1,size(T02f,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for fauty variables in normal data')
subplot(212)
plot(Tf2f,'k.-')
hold on
plot(ctr_Tnn *ones(1,size(Tf2f,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for fauty variables in fault data')
close all
for jj=1:size(Rsingfi{2},2)
Tfsing=DATAfiin{2}*Rsingfi{2}(:,1:jj);
Pfsing=(inv(Tfsing'*Tfsing)*Tfsing'*DATAfiin{2})';
Efii=DATAfiin{2}-DATAfiin{2}*Rsingfi{2}(:,1:jj)*Pfsing';
chaojie=find(abs(mean(Efii)-junzhif)>3*biaozhunchaf);
if size(chaojie,2)==0% if there is no point beyond 3*biaozhunchaf
    break
end
[Tf2t,speftemp]=online_monitoring(Efii,P0f,P0f,mean_Tnn ,invcov_Tnn );
figure
subplot(211)
plot(T02f,'k.-')
hold on
plot(ctr_Tnn *ones(1,size(T02f,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for fauty variables in normal data')
subplot(212)
plot(Tf2t,'k.-')
hold on
plot(ctr_Tnn *ones(1,size(Tf2t,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for fauty variables in fault data')
end

numj=5;
Tfsing=DATAfiin{2}*Rsingfi{2}(:,1:numj);
Pfsing=(inv(Tfsing'*Tfsing)*Tfsing'*DATAfiin{2})';
Efii=DATAfiin{2}-DATAfiin{2}*Rsingfi{2}(:,1:numj)*Pfsing';
[Tf2fr,speftemp]=online_monitoring(Efii,P0f,P0f,mean_Tnn ,invcov_Tnn );
figure
subplot(211)
plot(Tf2g,'k.-')
hold on
plot(ctr_Tn *ones(1,size(Tf2g,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for general variables in fault data')
subplot(212)
plot(Tf2fr,'k.-')
hold on
plot(ctr_Tnn *ones(1,size(Tf2fr,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for reconstructed fauty variables in fault data')

ffrec = ksdensity(T02f,Tf2fr);
figure
subplot(311)
plot(fg,'b.:')
xlabel('samples')
ylabel('probability')
title('For general variables')
subplot(312)
plot(ff,'k.:')
xlabel('samples')
ylabel('probability')
title('For faulty variables')
subplot(313)
plot(ffrec,'r.:')
xlabel('samples')
ylabel('probability')
title('For reconstructed faulty variables')


%fault reconstruction using fault models developed from other faults.
%indexfaultt2;Tf0g2;meand2,stdd2;ctr_Tn2;Tf0g2;meandf2.stddf2;P0f2;mean_Tnn2
%,invcov_Tnn2 ;
[DATA2tesele,DATA2teleft]=select_matrix(DATAi{2},indexfaultt2);
[DATA0tesele,DATA0teleft]=select_matrix(DATAi{1},indexfaultt2);
DATAnii{1}=DATA0teleft;DATAnii{2}=DATA2teleft;DATAsum=[DATA0teleft;DATA2teleft];
DATA2teleftn=autoscale(DATA2teleft,meand2,stdd2);
[Tf2g2,speftemp]=online_monitoring(DATA2teleftn,P02,P02 ,mean_Tn2 ,invcov_Tn2 );
figure
subplot(211)
plot(Tf0g2,'k.-')
hold on
plot(ctr_Tn2 *ones(1,size(Tf0g2,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for general variables in normal data')
subplot(212)
plot(Tf2g,'k.-')
hold on
plot(ctr_Tn2 *ones(1,size(Tf2g,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for general variables in fault data')
fg2 = ksdensity(Tf0g2,Tf2g2);
figure
plot(fg2,'k.:')
xlabel('samples')
ylabel('probability')




DATAfii{1}=DATA0tesele;DATAfii{2}=DATA2tesele;
DATAfiin{2}=autoscale(DATAfii{2},meandf2,stddf2);
[Tf2f2,speftemp]=online_monitoring(DATAfiin{2},P0f2,P0f2,mean_Tnn2 ,invcov_Tnn2 );
ff2= ksdensity(T02f2,Tf2f2);
figure
plot(ff2,'k.:')
xlabel('samples')
ylabel('probability')

figure
subplot(211)
plot(T02f2,'k.-')
hold on
plot(ctr_Tnn2 *ones(1,size(T02f2,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for fauty variables in normal data')
subplot(212)
plot(Tf2f2,'k.-')
hold on
plot(ctr_Tnn2 *ones(1,size(Tf2f2,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for fauty variables in fault data')


Efii=DATAfiin{2}-DATAfiin{2}*Rsingfi2{2}(:,1:numj)*Pfsing2';
[Tf2fr2,speftemp]=online_monitoring(Efii,P0f2,P0f2,mean_Tnn2 ,invcov_Tnn2 );
figure
subplot(211)
plot(Tf2g2,'k.-')
hold on
plot(ctr_Tn2 *ones(1,size(Tf2g2,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for general variables in fault data')
subplot(212)
plot(Tf2fr2,'k.-')
hold on
plot(ctr_Tnn2 *ones(1,size(Tf2fr2,1)),'r:')
xlabel('Samples')
ylabel('D^2')
title('for reconstructed fauty variables in fault data')

ffrec2 = ksdensity(T02f2,Tf2fr2);
figure
subplot(311)
plot(fg2,'b.:')
xlabel('samples')
ylabel('probability')
title('For general variables')
subplot(312)
plot(ff2,'k.:')
xlabel('samples')
ylabel('probability')
title('For faulty variables')
subplot(313)
plot(ffrec2,'r.:')
xlabel('samples')
ylabel('probability')
title('For reconstructed faulty variables')


mean(fg)
std(fg)
mean(fg2)
std(fg2)

 
mean(ffrec)
std(ffrec)
mean(ffrec2)
std(ffrec2)

