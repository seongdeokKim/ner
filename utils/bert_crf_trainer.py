from copy import deepcopy

import numpy as np
from seqeval.metrics import f1_score, accuracy_score, classification_report

import torch


class CrfTrainer:

    def __init__(self, config):
        self.config = config
        self.best_model = None
        self.best_loss = np.inf
        #self.best_accuracy = 0.0
        self.classification_report = None

    def check_best(self, model, performance, report):
        loss = float(performance)
        if loss <= self.best_loss: # If current epoch returns lower validation loss,
            self.best_loss = loss  # Update lowest validation loss.
            self.best_model = deepcopy(model.state_dict()) # Update best model weights.
            self.classification_report = report

        #accuracy = float(performance)
        #if accuracy >= self.best_accuracy:
        #    self.best_accuracy = accuracy
        #    self.best_model = deepcopy(model.state_dict())
        #self.classification_report = report

    def train(
            self,
            model, optimizer, scheduler,
            train_loader, valid_loader,
            index_to_tag,
            device,
    ):

        for epoch in range(self.config.n_epochs):

            # ========================================
            #               Training
            # ========================================

            # Put the model into training mode.
            model.train()
            # Reset the total loss for this epoch.
            total_tr_loss = 0

            for step, mini_batch in enumerate(train_loader):
                input_ids, labels = mini_batch['input_ids'], mini_batch['labels']
                input_ids, labels = input_ids.to(device), labels.to(device)
                attention_mask = mini_batch['attention_mask']
                attention_mask = attention_mask.to(device)


                # reset the gradients of all model parameters
                optimizer.zero_grad()

                # Take feed-forward
                outputs = model(
                    input_ids,
                    token_type_ids=None,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss, logits = outputs[0], outputs[1]

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # track train loss
                total_tr_loss += loss.item()

                # update parameters
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_tr_loss = total_tr_loss / len(train_loader)

            print('Epoch {} - loss={:.4e}'.format(
                epoch+1,
                avg_tr_loss
            ))

            # ========================================
            #               Validation
            # ========================================

            # Put the model into evaluation mode
            model.eval()

            total_val_loss, total_val_accuracy = 0, 0
            true_tags, pred_tags = [], []
            for step, mini_batch in enumerate(valid_loader):
                input_ids, labels = mini_batch['input_ids'], mini_batch['labels']
                input_ids, labels = input_ids.to(device), labels.to(device)
                attention_mask = mini_batch['attention_mask']
                attention_mask = attention_mask.to(device)
                _ = mini_batch['subwords_start_idxs']

                # Telling the model not to compute or store gradients,
                with torch.no_grad():
                    # Forward pass
                    outputs = model(
                        input_ids,
                        token_type_ids=None,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss, sequence_of_tags = outputs[0], outputs[1]

                # Calculate the accuracy for this batch of test sentences.
                total_val_loss += loss.mean().item()

                # Move logits and labels to CPU
                label_ids = labels.to('cpu')
                sequence_of_tags = sequence_of_tags.to('cpu')

                for p, t in zip(sequence_of_tags.tolist(), label_ids.tolist()):
                    pred_tag = []
                    for p_i, t_i in zip(p, t):
                        if index_to_tag[t_i] != "PAD":
                            pred_tag.append(index_to_tag[p_i])
                    pred_tags.append(pred_tag)
                for t in label_ids.tolist():
                    true_tag = []
                    for t_i in t:
                        if index_to_tag[t_i] != "PAD":
                            true_tag.append(index_to_tag[t_i])
                    true_tags.append(true_tag)

            avg_val_loss = total_val_loss / len(valid_loader)
            avg_val_acc = accuracy_score(pred_tags, true_tags)
            avg_val_f1_score = f1_score(pred_tags, true_tags)

            self.check_best(model, avg_val_loss, classification_report(pred_tags, true_tags))

            print('Validation - loss={:.4e} accuracy={:.4f} f1-score={:.4f} best_loss={:.4f}'.format(
                avg_val_loss,
                avg_val_acc,
                avg_val_f1_score,
                self.best_loss,
            ))

        print(self.classification_report)
        model.load_state_dict(self.best_model)

        return model