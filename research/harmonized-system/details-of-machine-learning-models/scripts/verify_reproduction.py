"""Refit in an isolated new directory and compare all saved numerical outputs."""
from pathlib import Path
import tempfile,shutil,json
import analyze_models as analysis
from threadpoolctl import threadpool_limits

def main():
    root=Path(__file__).resolve().parents[1]
    directory=Path(tempfile.mkdtemp(prefix='chapter-reproduction-',dir=root/'results')).resolve()
    try:
        shutil.copyfile(root/'toy_trade_dataset.csv',directory/'toy_trade_dataset.csv')
        analysis.ROOT=directory
        with threadpool_limits(limits=2):analysis.run()
        names=['frozen_2025_predictions.csv','predictions.csv','metrics.json','worked_examples.json','lambda_estimation.csv','elastic_tuning.json','fit_records.json']
        checks=[]
        for name in names:
            expected=analysis.sha(root/'results'/name);actual=analysis.sha(directory/'results'/name)
            assert actual==expected,'Non-identical reproduction: '+name
            checks.append({'file':name,'sha256':actual,'bit_identical':True})
        (root/'results/reproduction_verification.json').write_text(json.dumps({'status':'PASS','independent_refit':True,'checks':checks},indent=2),encoding='utf-8')
        print('PASS: independent refit produced bit-identical forecasts, all predictions, metrics, traces, lambda, tuning and fit records.',flush=True)
    finally:
        # This temporary path was created here and must remain within the new workspace project.
        assert directory.parent==(root/'results').resolve() and directory.name.startswith('chapter-reproduction-')
        shutil.rmtree(directory)
if __name__=='__main__':main()
