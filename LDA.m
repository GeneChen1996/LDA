clc;       % 清除command window
clear      % 清除workspace
close all  % 關閉所有figure
%%-----------------------------------------------------------------------%%
dataSet = load('iris.txt');
rawData = dataSet(51:150,[3,4]);  %輸入資料class3,4 特徵

%%--------訓練與測試集的標籤
train_label1 = [dataSet(51:75,5);...
              dataSet(101:125,5)];

test_label1 = [dataSet(76:100,5);...
              dataSet(126:150,5)];

          
%%--------class中前25筆data設為training data，剩餘的25筆設為test data

trainset1 = [rawData(1:25,1:2);...
            rawData(51:75,1:2)];
testdata1 = [rawData(26:50,1:2);...
            rawData(76:100,1:2)];
        
%%--------設定positive與negative類別至訓練集  
trainsetP1 = trainset1(1:25,1:2);
trainsetN1 = trainset1(26:50,1:2);   
trainsetP1 = trainsetP1';
trainsetN1 = trainsetN1';

%%--------轉置訓練集以利於計算        
trainset1 = trainset1';
testdata1 = testdata1';
        
%%----------------------開始計算weight與bias-----------------------------%%

mu1 = mean(trainset1,2);  %訓練集平均
mu11 = mean(trainsetP1,2);  %正類別訓練集平均
mu12 = mean(trainsetN1,2);  %負別訓練集平均

d1 = trainset1 - repmat(mu1,1,50);  %訓練集所有值減去訓練集平均值

%%--------invsigma與平均值互減轉置的計算
s1 = d1*d1';
sw1 = s1*(1/(50-1));
invsigma1 = inv(sw1);

sb1 = (mu11-mu12)';
wT1 = sb1*invsigma1;  %得出權重的轉置

%%--------bias計算
C12=1;
C21=1;
pi1 = 0.5;
pi2 = 0.5;

b1 = -0.5*sb1*invsigma1*(mu11+mu12)-log((C12*pi2)/(C21*pi1));

%%--------將測試集輸入至所訓練LDA進行分類

trainset1_result1 = wT1 * trainset1 + b1;

output_train_class1 = (trainset1_result1 < 0)+2;  %輸出訓練集所分類出標籤
output_train_class1 = output_train_class1';  %轉置訓練集以利於比對

testset_result1 = wT1 * testdata1 + b1;

output_test_class1 = (testset_result1 < 0)+2;  %輸出測試集所分類出標籤
output_test_class1 = output_test_class1'; %轉置測試集以利於比對


%%--------比對標籤後得出分類率
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
%%-------------------------訓練集與測試集互換--------------------------%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%--------訓練與測試集的標籤

train_label2 = [dataSet(76:100,5);...
              dataSet(126:150,5)];

test_label2 = [dataSet(51:75,5);...
              dataSet(101:125,5)];

          
%%--------class中後25筆data設為training data，剩餘的25筆設為test data

trainset2 = [rawData(26:50,1:2);...
            rawData(76:100,1:2)];
testdata2 = [rawData(1:25,1:2);...
            rawData(51:75,1:2)];
        
%%--------設定positive與negative類別至訓練集  
trainsetP2 = trainset2(1:25,1:2);
trainsetN2 = trainset2(26:50,1:2);   
trainsetP2 = trainsetP2';
trainsetN2 = trainsetN2';

%%--------轉置訓練集以利於計算        
trainset2 = trainset2';
testdata2 = testdata2';
        
%%----------------------開始計算weight與bias-----------------------------%%

mu2 = mean(trainset2,2);  %訓練集平均
mu21 = mean(trainsetP2,2);  %正類別訓練集平均
mu22 = mean(trainsetN2,2);  %負別訓練集平均

d2 = trainset2 - repmat(mu2,1,50);  %訓練集所有值減去訓練集平均值

%%--------invsigma與平均值互減轉置的計算
s2 = d2*d2';
sw2 = s2*(1/(50-1));
invsigma2 = inv(sw2);

sb2 = (mu21-mu22)';
wT2 = sb2*invsigma2;  %得出權重的轉置

%%--------bias計算

b2 = -0.5*sb2*invsigma2*(mu21+mu22)-log((C12*pi2)/(C21*pi1));

%%--------將測試集輸入至所訓練LDA進行分類
trainset2_result2 = wT2 * trainset2 + b2;

output_train_class2 = (trainset2_result2 < 0)+2;  %輸出訓練集所分類出標籤
output_train_class2 = output_train_class2';  %轉置訓練集以利於比對


testset_result2 = wT2 * testdata2 + b2;

output_test_class2 = (testset_result2 < 0)+2;  %輸出訓練集所分類出標籤
output_test_class2 = output_test_class2';  %轉置訓練集以利於比對

%%--------比對標籤後得出分類率
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
    

%%-----------------------------結果-----------------------------------%%


%%--------畫出訓練集散佈圖與hyperplane
trainset1 = trainset1';  %將訓練集轉置回原本矩陣以利於畫圖
trainset2 = trainset2';

figure;
h=gscatter(trainset1(:,1),trainset1(:,2),train_label1,'rb','v^',[],'off')
hold on

title('訓練集分類散佈圖1');  % 圖名稱
legend('classP1', 'classN1');  % 類別標號說明
xlabel('Feature3');  % 特徵標號註解
ylabel('Feature4');

%%--------畫出LDA
lx1 = get(gca, 'Xlim');
ly1 =-wT1(1)/wT1(2)*lx1-b1/wT1(2);
plot(lx1, ly1, '-g', 'DisplayName', 'LDA1')
xlim([-inf inf]) 
ylim([-inf inf])
hold off


%%--------------訓練集調換後之訓練集散佈圖與hyperplane
figure;
h2=gscatter(trainset2(:,1),trainset2(:,2),train_label2,'rb','ox',[],'off')
hold on

title('訓練集分類散佈圖2');  % 圖名稱
legend('classP2', 'classN2');  % 類別標號說明
xlabel('Feature3');  % 特徵標號註解
ylabel('Feature4');

%%--------畫出LDA
lx2 = get(gca, 'Xlim');
ly2 =-wT2(1)/wT2(2)*lx2-b2/wT2(2);
plot(lx2, ly2, '-g', 'DisplayName', 'LDA2')
xlim([-inf inf]) 
ylim([-inf inf])
hold off


fprintf('hyperplane 之 weight vectorm 與 bias\n');
wT1
b1
wT2
b2


fprintf('第一次分類率CR1 = %2.4f%%\n', CR1*100);
fprintf('第二次分類率CR2 = %2.4f%%\n', CR2*100);

CR = (CR1+CR2)/2;
fprintf('平均分類率CR = %2.4f%%\n', CR*100);






