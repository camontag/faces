%----------------------------------------------------------------
% File:     main_fitcecoc.m
%----------------------------------------------------------------
%
% Author:   Marek Rychlik (rychlik@arizona.edu)
% Date:     Fri Nov 22 20:02:05 2024
% Copying:  (C) Marek Rychlik, 2020. All rights reserved.
% 
%----------------------------------------------------------------
% Classification into several classes
% This script trains a facial recognition model. The model
% is saved to a .MAT file, along with necessary data to perform facial
% recognition:

targetSize = [128,128];
k=60;                                   % Number of features to consider
location = fullfile('lfw');

disp('Creating image datastore...');
imds0 = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));

disp('Creating subset of several persons...');
persons = {'Angelina_Jolie', 'Eduardo_Duhalde', 'Amelie_Mauresmo', 'Abdullah_Gul',...
    'Adrien_Brody', 'Al_Gore', 'Albert_Costa', 'Alejandro_Toledo', 'Ali_Naimi',...
    'Alvaro_Uribe', 'Ana_Palacio', 'Andre_Agassi', 'Andy_Roddick', 'Ann_Veneman',...
    'Anna_Kournikova', 'Ariel_Sharon', 'Arnold_Schwarzenegger', 'Atal_Bihari_Vajpayee',...
    'Ben_Affleck', 'Bill_Clinton', 'Bill_Gates', 'Bill_Simon', 'Britney_Spears', ...
    'Carlos_Menem', 'Carlos_Moya', 'Catherine_Zeta-Jones', 'Charles_Moose', ...
    'Colin_Powell', 'David_Nalbandian', 'Dick_Cheney', 'Dominique_de_Villepin',...
    'Donald_Rumsfeld', 'Edmund_Stoiber', 'Fidel_Castro', 'George_HW_Bush',...
    'George_Robertson', 'Gerhard_Schroeder', 'Gloria_Macapagal_Arroyo', ...
    'Gonzalo_Sanchez_de_Lozada', 'Gordon_Brown', 'Gray_Davis', 'Guillermo_Coria', ...
    'Halle_Berry', 'Hamid_Karzai', 'Hans_Blix', 'Hillary_Clinton', 'Hu_Jintao',...
    'Hugo_Chavez', 'Ian_Thorpe', 'Igor_Ivanov', 'Jacques_Chirac', 'Jean_Chretien',...
    'Jeb_Bush', 'Jennifer_Aniston', 'Jennifer_Capriati', 'Jennifer_Garner',...
    'Jennifer_Lopez', 'Jeremy_Greenstock', 'Jiang_Zemin', 'John_Allen_Muhammad', 'John_Ashcroft',...
    'John_Bolton', 'John_Howard', 'John_Kerry', 'John_Negroponte', 'John_Snow', ...
    'Joschka_Fischer', 'Jose_Maria_Aznar', 'Juan_Carlos_Ferrero', 'Julianne_Moore', ...
    'Junichiro_Koizumi', 'Kim_Jong-Il', 'Kofi_Annan', 'Lance_Armstrong', 'Laura_Bush', ...
    'Lindsay_Davenport', 'Lleyton_Hewitt', 'Lucio_Gutierrez', 'Luiz_Inacio_Lula_da_Silva', ...
    'Mahmoud_Abbas', 'Mark_Philippoussis', 'Megawati_Sukarnoputri', 'Meryl_Streep', ...
    'Michael_Bloomberg', 'Michael_Jackson', 'Michael_Schumacher', 'Mike_Weir', 'Mohammad_Khatami',...
    'Mohammed_Al-Douri', 'Muhammad_Ali', 'Nancy_Pelosi', 'Naomi_Watts', 'Nestor_Kirchner', ...
    'Nicanor_Duarte_Frutos', 'Nicole_Kidman', 'Norah_Jones', 'Paradorn_Srichaphan', ...
    'Paul_Bremer', 'Paul_Wolfowitz', 'Pervez_Musharraf', 'Pete_Sampras', 'Pierce_Brosnan',...
    'Queen_Elizabeth_II', 'Recep_Tayyip_Erdogan', 'Renee_Zellweger', 'Ricardo_Lagos', 'Richard_Gephardt', ...
    'Richard_Myers', 'Roger_Federer', 'Roh_Moo-hyun', 'Rubens_Barrichello', 'Rudolph_Giuliani', ...
    'Saddam_Hussein', 'Salma_Hayek', 'Serena_Williams', 'Sergey_Lavrov', 'Silvio_Berlusconi', ...
    'Spencer_Abraham', 'Taha_Yassin_Ramadan', 'Tang_Jiaxuan', 'Tiger_Woods'};
persons = persons.';
persons = categorical(persons);
[lia, locb] = ismember(imds0.Labels, persons);
imds = subset(imds0, lia);

t=tiledlayout('flow');
nexttile(t);
montage(imds);

disp('Reading all images');
A = readall(imds);

B = cat(3,A{:});
D = prod(targetSize);
B = reshape(B,D,[]);

disp('Normalizing data...');
B = single(B)./256;
% NOTE: Normalization subtracts the mean pixel value
% from all pixels and divides by standard deviation. It is
% equivalent to:
%     [B,C,SD] = normalize(B, 1)
% This procedure is different from an alternative:
%     [B,C,SD] = normalize(B, 2)
% which computes the 'mean face' and subtracts it from every
% face. SD is then the l^2-norm between a face and mean face.
[B,C,SD] = normalize(B);
tic;
[U,S,V] = svd(B,'econ');
toc;

% Get an montage of eigenfaces
Eigenfaces = arrayfun(@(j)reshape((U(:,j)-min(U(:,j)))./(max(U(:,j))-min(U(:,j))),targetSize), ...
    1:size(U,2),'uni',false);

nexttile(t);
montage(Eigenfaces(1:16));
title('Top 16 Eigenfaces');
colormap(gray);

% NOTE: Rows of V are observations, columns are features.
% Observations need to be in rows.
k = min(size(V,2),k);

% Discard unnecessary data
W = S * V';                             % Transform V to weights (ala PCA)
W = W(1:k,:);                           % Keep first K weights
% NOTE: We will never again need singular values S
%S = diag(S);
%S = S(1:k);
U = U(:,1:k);                           % Keep K eigenfaces

% Find feature vectors of all images
X = W';
Y = categorical(imds.Labels, persons);

% Create colormap
cm=[1,0,0;
    0,0,1,
    0,1,0];
% Assign colors to target values
c=cm(1+mod(uint8(Y),size(cm,1)),:);

disp('Training Support Vector Machine...');
options = statset('UseParallel',true);
tic;

% You may try this, to get a more optimized model
% 'OptimizeHyperparameters','all',...

Mdl = fitcecoc(X, Y,'Verbose', 2,'Learners','svm',...
               'Options',options, 'OptimizeHyperparameters','all');
toc;

% Generate a plot in feature space using top two features
% nexttile(t);
% scatter3(X(:,1),X(:,2),X(:,3),50,c);
% title('A top 3-predictor plot');
% xlabel('x1');
% ylabel('x2');
% zlabel('x3');
% 
% nexttile(t);
% scatter3(X(:,4),X(:,5),X(:,6),50,c);
% title('A next 3-predictor plot');
% xlabel('x4');
% ylabel('x5');
% zlabel('x6');

%[YPred,Score] = predict(Mdl,X);
[YPred,Score,Cost] = resubPredict(Mdl);

% ROC = receiver operating characteristic
% See https://en.wikipedia.org/wiki/Receiver_operating_characteristic
% disp('Plotting ROC metrics...');
% rm = rocmetrics(imds.Labels, Score, persons);
% nexttile(t);
% plot(rm);

disp('Plotting confusion matrix...')
nexttile(t);
confusionchart(Y, YPred);
title(['Number of features: ' ,num2str(k)]);

% Save the model and persons that the model recognizes.
% NOTE: An important part of the submission.
save('model','Mdl','persons','U', 'targetSize');
