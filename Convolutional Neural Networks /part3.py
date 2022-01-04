import argparse
import io
import sys
import re

class CapturedOutput():
    # context manager to capture stdout
    def __init__(self):
        self.output = ''

    def __enter__(self):
        self.capturedOutput = io.StringIO()
        sys.stdout = self.capturedOutput
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.output = self.capturedOutput.getvalue()
        sys.stdout = sys.__stdout__

def part3():
    from eval_miniplaces import main as test
    from train_miniplaces import main as train
    batch = [32, 8, 16, 32, 32, 32, 32]
    learn = [0.001, 0.001, 0.001, 0.05, 0.01, 0.001, 0.001]
    epoch = [10, 10, 10, 10, 10, 20, 5]
    results = []
    for i, (b, l, e) in enumerate(zip(batch, learn, epoch)):
        print('*************************************************************************')
        print(f'Test {i+1}/7 using {e} epochs, {l} learning rate, {b} batch size')
        print('*************************************************************************')
        train(argparse.Namespace(epochs=e, lr=l, batch_size=b, resume=''))
        with CapturedOutput() as c:
            test(argparse.Namespace(load='./outputs/model_best.pth.tar'))
        print(c.output)
        results.append(re.search('Accuracy: (\d+\.\d+)%', c.output)[1])
    with open('results.txt', 'w+') as f:
        f.write('\n'.join(results))

if __name__ == '__main__':
    part3()