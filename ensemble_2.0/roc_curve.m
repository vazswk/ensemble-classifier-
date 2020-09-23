% Usage:
% model = params;
% [predict_label,mse,deci] = svmpredict(y,x,model); % the procedure for predicting
% auc = roc_curve(deci*model.Label(1),y);
function [auc,stack_x,stack_y] = roc_curve(deci,label_y)
	[~,ind] = sort(deci,'descend');
	roc_y = label_y(ind);
	stack_x = cumsum(roc_y == -1)/sum(roc_y == -1);
	stack_y = cumsum(roc_y == 1)/sum(roc_y == 1);
	auc = sum((stack_x(2:length(roc_y),1)-stack_x(1:length(roc_y)-1,1)).*stack_y(2:length(roc_y),1))

    %Comment the above lines if using perfcurve of statistics toolbox
%     [stack_x,stack_y,thre,auc]=perfcurve(label_y,deci,1);
    figure;
%     stack_x = stack_x(1:10:end); % smooth the ROC curve
%     stack_y = stack_y(1:10:end);
	plot(stack_x,stack_y);
	xlabel('False Positive Rate');
	ylabel('True Positive Rate');
	title(['ROC curve of (AUC = ' num2str(auc) ' )']);
end