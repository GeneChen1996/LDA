clc;       % �M��command window
clear      % �M��workspace
close all  % �����Ҧ�figure
%%-----------------------------------------------------------------------%%
dataSet = load('iris.txt');
rawData = dataSet(51:150,[3,4]);  %��J���class3,4 �S�x

%%--------�V�m�P���ն�������
train_label1 = [dataSet(51:75,5);...
              dataSet(101:125,5)];

test_label1 = [dataSet(76:100,5);...
              dataSet(126:150,5)];

          
%%--------class���e25��data�]��training data�A�Ѿl��25���]��test data

trainset1 = [rawData(1:25,1:2);...
            rawData(51:75,1:2)];
testdata1 = [rawData(26:50,1:2);...
            rawData(76:100,1:2)];
        
%%--------�]�wpositive�Pnegative���O�ܰV�m��  
trainsetP1 = trainset1(1:25,1:2);
trainsetN1 = trainset1(26:50,1:2);   
trainsetP1 = trainsetP1';
trainsetN1 = trainsetN1';

%%--------��m�V�m���H�Q��p��        
trainset1 = trainset1';
testdata1 = testdata1';
        
%%----------------------�}�l�p��weight�Pbias-----------------------------%%

mu1 = mean(trainset1,2);  %�V�m������
mu11 = mean(trainsetP1,2);  %�����O�V�m������
mu12 = mean(trainsetN1,2);  %�t�O�V�m������

d1 = trainset1 - repmat(mu1,1,50);  %�V�m���Ҧ��ȴ�h�V�m��������

%%--------invsigma�P�����Ȥ�����m���p��
s1 = d1*d1';
sw1 = s1*(1/(50-1));
invsigma1 = inv(sw1);

sb1 = (mu11-mu12)';
wT1 = sb1*invsigma1;  %�o�X�v������m

%%--------bias�p��
C12=1;
C21=1;
pi1 = 0.5;
pi2 = 0.5;

b1 = -0.5*sb1*invsigma1*(mu11+mu12)-log((C12*pi2)/(C21*pi1));

%%--------�N���ն���J�ܩҰV�mLDA�i�����

trainset1_result1 = wT1 * trainset1 + b1;

output_train_class1 = (trainset1_result1 < 0)+2;  %��X�V�m���Ҥ����X����
output_train_class1 = output_train_class1';  %��m�V�m���H�Q����

testset_result1 = wT1 * testdata1 + b1;

output_test_class1 = (testset_result1 < 0)+2;  %��X���ն��Ҥ����X����
output_test_class1 = output_test_class1'; %��m���ն��H�Q����


%%--------�����ҫ�o�X�����v
PCR1= length(find(output_test_class1(1:25,:) == test_label1(1:25,:)))/25;
NCR1= length(find(output_test_class1(26:50,:) == test_label1(26:50,:)))/25;
CR1 = length(find(output_test_class1(1:50,:) == test_label1(1:50,:)))/50;

TP1 = length(find(output_test_class1(1:25,:) == test_label1(1:25,:)));
FP1 = 25-length(find(output_test_class1(1:25,:) == test_label1(1:25,:)));

FN1 = 25-length(find(output_test_class1(26:50,:) == test_label1(26:50,:)));
TN1 = length(find(output_test_class1(26:50,:) == test_label1(26:50,:)));

Confusionmatrix1=[TP1 FP1; FN1 TN1]


fpr1 = FP1/(FP1+TN1);
fpr1(isnan(fpr1))= 0;
tpr1 = TP1/(TP1+FN1);
tpr1(isnan(tpr1))= 0;





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%-------------------------�V�m���P���ն�����--------------------------%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%--------�V�m�P���ն�������

train_label2 = [dataSet(76:100,5);...
              dataSet(126:150,5)];

test_label2 = [dataSet(51:75,5);...
              dataSet(101:125,5)];

          
%%--------class����25��data�]��training data�A�Ѿl��25���]��test data

trainset2 = [rawData(26:50,1:2);...
            rawData(76:100,1:2)];
testdata2 = [rawData(1:25,1:2);...
            rawData(51:75,1:2)];
        
%%--------�]�wpositive�Pnegative���O�ܰV�m��  
trainsetP2 = trainset2(1:25,1:2);
trainsetN2 = trainset2(26:50,1:2);   
trainsetP2 = trainsetP2';
trainsetN2 = trainsetN2';

%%--------��m�V�m���H�Q��p��        
trainset2 = trainset2';
testdata2 = testdata2';
        
%%----------------------�}�l�p��weight�Pbias-----------------------------%%

mu2 = mean(trainset2,2);  %�V�m������
mu21 = mean(trainsetP2,2);  %�����O�V�m������
mu22 = mean(trainsetN2,2);  %�t�O�V�m������

d2 = trainset2 - repmat(mu2,1,50);  %�V�m���Ҧ��ȴ�h�V�m��������

%%--------invsigma�P�����Ȥ�����m���p��
s2 = d2*d2';
sw2 = s2*(1/(50-1));
invsigma2 = inv(sw2);

sb2 = (mu21-mu22)';
wT2 = sb2*invsigma2;  %�o�X�v������m

%%--------bias�p��

b2 = -0.5*sb2*invsigma2*(mu21+mu22)-log((C12*pi2)/(C21*pi1));

%%--------�N���ն���J�ܩҰV�mLDA�i�����
trainset2_result2 = wT2 * trainset2 + b2;

output_train_class2 = (trainset2_result2 < 0)+2;  %��X�V�m���Ҥ����X����
output_train_class2 = output_train_class2';  %��m�V�m���H�Q����


testset_result2 = wT2 * testdata2 + b2;

output_test_class2 = (testset_result2 < 0)+2;  %��X�V�m���Ҥ����X����
output_test_class2 = output_test_class2';  %��m�V�m���H�Q����

%%--------�����ҫ�o�X�����v
PCR2= length(find(output_test_class2(1:25,:) == test_label2(1:25,:)))/25;
NCR2= length(find(output_test_class2(26:50,:) == test_label2(26:50,:)))/25;
CR2 = length(find(output_test_class2(1:50,:) == test_label2(1:50,:)))/50;

TP2 = length(find(output_test_class2(1:25,:) == test_label2(1:25,:)));
FP2 = 25-length(find(output_test_class2(1:25,:) == test_label2(1:25,:)));

FN2 = 25-length(find(output_test_class2(26:50,:) == test_label2(26:50,:)));
TN2 = length(find(output_test_class2(26:50,:) == test_label2(26:50,:)));

Confusionmatrix2=[TP2 FP2; FN2 TN2]

fpr2 = FP2/(FP2+TN2);
fpr2(isnan(fpr2))= 0;
tpr2 = TP2/(TP2+FN2);
tpr2(isnan(tpr2))= 0;
    

%%-----------------------------���G-----------------------------------%%


%%--------�e�X�V�m�����G�ϻPhyperplane
trainset1 = trainset1';  %�N�V�m����m�^�쥻�x�}�H�Q��e��
trainset2 = trainset2';

figure;
h=gscatter(trainset1(:,1),trainset1(:,2),train_label1,'rb','v^',[],'off')
hold on

title('�V�m���������G��1');  % �ϦW��
legend('classP1', 'classN1');  % ���O�и�����
xlabel('Feature3');  % �S�x�и�����
ylabel('Feature4');

%%--------�e�XLDA
lx1 = get(gca, 'Xlim');
ly1 =-wT1(1)/wT1(2)*lx1-b1/wT1(2);
plot(lx1, ly1, '-g', 'DisplayName', 'LDA1')
xlim([-inf inf]) 
ylim([-inf inf])
hold off


%%--------------�V�m���մ��ᤧ�V�m�����G�ϻPhyperplane
figure;
h2=gscatter(trainset2(:,1),trainset2(:,2),train_label2,'rb','ox',[],'off')
hold on

title('�V�m���������G��2');  % �ϦW��
legend('classP2', 'classN2');  % ���O�и�����
xlabel('Feature3');  % �S�x�и�����
ylabel('Feature4');

%%--------�e�XLDA
lx2 = get(gca, 'Xlim');
ly2 =-wT2(1)/wT2(2)*lx2-b2/wT2(2);
plot(lx2, ly2, '-g', 'DisplayName', 'LDA2')
xlim([-inf inf]) 
ylim([-inf inf])
hold off


fprintf('hyperplane �� weight vectorm �P bias\n');
wT1
b1
wT2
b2


fprintf('�Ĥ@�������vCR1 = %2.4f%%\n', CR1*100);
fprintf('�ĤG�������vCR2 = %2.4f%%\n', CR2*100);

CR = (CR1+CR2)/2;
fprintf('���������vCR = %2.4f%%\n', CR*100);






