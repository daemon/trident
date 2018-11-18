import argparse

from tqdm import tqdm
import argconf
import torch
import torch.utils.data as td

import deeplm


def train(config):
    vocab, (train_ds, dev_ds, test_ds) = deeplm.data.Seq2SeqDataset.iters(config)
    model = deeplm.model.InterstitialModel(vocab, config)
    model.train()
    train_loader = td.DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=train_ds.collate)
    dev_loader = td.DataLoader(dev_ds, batch_size=config["batch_size"], collate_fn=dev_ds.collate)
    test_loader = td.DataLoader(test_ds, batch_size=config["batch_size"], collate_fn=test_ds.collate)
    model.cuda()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 3, 1]).cuda(), ignore_index=vocab.tok2idx[deeplm.data.PAD_TOKEN])
    step_no = 1
    dev_step_no = 1
    min_dev_loss = 10000
    if config["resume"]:
        sd = torch.load(config["resume"])
        model.load_state_dict(sd["state"])
        model.avg_param = sd["ema"]
        model.steps_ema = sd["steps_ema"]
        min_dev_loss = sd["loss"]
        config["epochs"] -= sd["epoch"]
    if config.get("test"):
        config["epochs"] = 1
    print("epoch,step,type,loss,pos_acc,neg_acc")
    for idx in tqdm(range(config["epochs"]), position=0):
        pbar = tqdm(total=len(train_loader), position=1)
        pbar.set_description("Training")
        model.train()
        if idx == 8:
            optimizer.param_groups[0]["lr"] = config["lr"] / 10
        if not config.get("test"):
            for transcripts, targets, lengths in train_loader:
                optimizer.zero_grad()
                transcripts = transcripts.cuda()
                lengths = lengths.cuda()
                targets = targets.cuda()
                scores = model(transcripts, lengths, gt=targets)
                loss = criterion(scores, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, config["clip_grad_norm"])
                optimizer.step()

                pos_acc = (scores.max(1)[1] == targets)[targets == 1].float().mean()
                neg_acc = (scores.max(1)[1] == targets)[targets == 0].float().mean()
                model.update_ema()
                print(",".join((str(idx), str(step_no), "train", f"{loss.item():.4f}", f"{100 * pos_acc:.2f}", f"{100 * neg_acc:.2f}")))
                pbar.set_postfix(dict(loss=f"{loss.item():.3f}", acc=f"{pos_acc.item():.2f}"), neg_acc=f"{neg_acc.item():.2f}")
                pbar.update(1)
                step_no += 1
            pbar.close()

        for loader, name in zip((dev_loader, test_loader), ("dev", "test")):
            pbar = tqdm(total=len(loader), position=1)
            pbar.set_description(name.capitalize())
            dev_loss = 0
            avg = 0
            model.eval()
            ints = ["X", "-", ""]
            params = model.get_params()
            model.load_ema_params()
            for transcripts, targets, lengths in loader:
                transcripts = transcripts.cuda()
                lengths = lengths.cuda()
                targets = targets.cuda()
                scores = model(transcripts, lengths, gt=targets)
                loss = criterion(scores, targets)
                dev_loss += loss.item()
                avg += targets.size(0)
                pos_acc = (scores.max(1)[1] == targets)[targets == 1].float().mean()
                neg_acc = (scores.max(1)[1] == targets)[targets == 0].float().mean()
                if config.get("test"):
                    print("".join([vocab.idx2tok[idx] for idx in transcripts[0].cpu().numpy().tolist()]))
                    print("".join([vocab.idx2tok[idx] for idx in targets[0].cpu().numpy().tolist()]))
                    print("".join([ints[idx] for idx in scores.max(1)[1][0].cpu().numpy().tolist()]))
                    print()
                print(",".join((str(idx), str(dev_step_no), name, f"{loss.item():.4f}", f"{100 * pos_acc:.2f}", f"{100 * neg_acc:.2f}")))
                dev_step_no += 1
                pbar.set_postfix(dict(loss=f"{loss.item():.3f}", acc=f"{pos_acc.item():.2f}"), neg_acc=f"{neg_acc.item():.2f}")
                pbar.update(1)
            model.load_params(params)
            if dev_loss / avg < min_dev_loss and name == "dev" and not config.get("test"):
                min_dev_loss = dev_loss / avg
                torch.save(dict(loss=min_dev_loss, epoch=idx + 1, state=model.state_dict(), ema=model.avg_param, steps_ema=model.steps_ema), config["output_file"])
            pbar.close()


def main():
    description = "Trains or evaluates a model."
    epilog = "Usage:\npython -m deeplm.utils.run -c confs/default.json --action train"
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("-c", "--config", dest="config", type=str, default="confs/default.json")
    parser.add_argument("--action", choices=["train", "eval"], type=str, default="train")
    args = parser.parse_args()
    conf = argconf.config_from_json(args.config)
    if args.action == "train":
        train(conf)


if __name__ == "__main__":
    main()
