import { useEffect, useState } from 'react';
import { Upload, ShieldCheck, Eye, AlertTriangle, Car, Activity, Info, BrainCircuit } from 'lucide-react';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [system, setSystem] = useState({
    detector_model: 'loading',
    visibility_model: 'loading',
  });

  useEffect(() => {
    let ignore = false;

    fetch(API + '/api/model-status')
      .then((r) => r.json())
      .then((d) => {
        if (!ignore) setSystem(d);
      })
      .catch(() => {});

    return () => {
      ignore = true;
    };
  }, []);

  const choose = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;

    setFile(f);
    setResult(null);
    setError('');

    if (f.type.startsWith('image/')) {
      setPreview(URL.createObjectURL(f));
    } else {
      setPreview('');
    }
  };

  const analyze = async () => {
    if (!file) return;

    setLoading(true);
    setError('');

    const fd = new FormData();
    fd.append('file', file);

    try {
      const r = await fetch(API + '/api/analyze', {
        method: 'POST',
        body: fd,
      });
      const d = await r.json();
      if (!r.ok) throw Error(d.detail || 'Analysis failed');
      setResult(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const media = result ? API + result.annotated_file : '';

  return (
    <div className="app">
      <header>
        <div className="brand">
          <div className="logo">
            <ShieldCheck />
          </div>
          <div>
            <h1>Mine Vehicle Safety AI</h1>
            <p>Fog &amp; Low-Visibility Risk Monitoring</p>
          </div>
        </div>
        <div className="status">
          <i /> {system.detector_model === 'trained' ? 'TRAINED MODEL READY' : 'DEMO MODE'}
        </div>
      </header>

      <main>
        <section className="hero">
          <div>
            <span>AIML DRIVER-ASSISTANCE PLATFORM</span>
            <h2>Safer mine vehicle operation in difficult visibility.</h2>
            <p>
              Upload a road-facing image or video. The system detects objects, estimates visibility,
              and produces an explainable risk warning using either a trained model or a fallback
              pretrained demo pipeline.
            </p>
          </div>
          <div className="heroIcon">
            <BrainCircuit size={64} />
          </div>
        </section>

        <section className="grid">
          <div className="panel">
            <div className="title">
              <Upload /> Input
            </div>

            <label className="drop">
              <input type="file" accept="image/*,video/*" onChange={choose} />
              <Upload size={32} />
              <strong>{file ? file.name : 'Choose image or video'}</strong>
              <small>JPG, PNG, WEBP, MP4, AVI, MOV</small>
            </label>

            {preview && <img className="preview" src={preview} alt="Selected upload preview" />}

            <button onClick={analyze} disabled={!file || loading}>
              {loading ? 'Analyzing...' : 'Analyze with AI'}
            </button>

            {error && <div className="error">{error}</div>}
          </div>

          <div className="panel">
            <div className="title">
              <Activity /> Analysis Result
            </div>

            {!result ? (
              <div className="empty">
                <Eye size={38} />
                <p>Upload media to see AI analysis.</p>
              </div>
            ) : (
              <>
                <div
                  className={
                    'risk ' +
                    (result.risk_level.includes('HIGH')
                      ? 'high'
                      : result.risk_level === 'CAUTION'
                        ? 'warn'
                        : 'safe')
                  }
                >
                  <AlertTriangle />
                  <div>
                    <small>RISK LEVEL</small>
                    <b>{result.risk_level}</b>
                    <span>{result.warning}</span>
                  </div>
                </div>

                <div className="metrics">
                  <div>
                    <span>Visibility</span>
                    <b>{result.visibility_score}%</b>
                    <em>{result.visibility_status}</em>
                  </div>
                  <div>
                    <span>Objects</span>
                    <b>{result.object_count}</b>
                    <em>detected</em>
                  </div>
                  <div>
                    <span>Frames</span>
                    <b>{result.frames_analyzed}</b>
                    <em>analyzed</em>
                  </div>
                </div>
              </>
            )}
          </div>
        </section>

        <section className="mini-grid">
          <div className="panel compact">
            <div className="title">
              <Car /> Model mode
            </div>
            <div className="mode-row">
              <span>Detector</span>
              <strong>{system.detector_model}</strong>
            </div>
            <div className="mode-row">
              <span>Visibility</span>
              <strong>{system.visibility_model}</strong>
            </div>
          </div>

          <div className="panel compact">
            <div className="title">
              <BrainCircuit /> AIML workflow
            </div>
            <ul className="workflow-list">
              <li>Dataset preparation</li>
              <li>Preprocessing &amp; feature extraction</li>
              <li>Training &amp; validation</li>
              <li>Testing &amp; evaluation</li>
              <li>Inference and deployment</li>
            </ul>
          </div>
        </section>

        {result && (
          <section className="grid">
            <div className="panel">
              <div className="title">
                <Eye /> AI Output
              </div>

              {result.media_type === 'image' ? (
                <img className="resultMedia" src={media} alt="AI annotated output" />
              ) : (
                <video className="resultMedia" src={media} controls />
              )}
            </div>

            <div className="panel">
              <div className="title">
                <Car /> Detected Objects
              </div>

              <div className="table">
                <div className="row head">
                  <span>Object</span>
                  <span>Confidence</span>
                  <span>Risk</span>
                </div>

                {result.detections.map((d, i) => (
                  <div className="row" key={i}>
                    <span>{d.label}</span>
                    <span>{(d.confidence * 100).toFixed(1)}%</span>
                    <span className={'tag ' + d.risk.toLowerCase()}>{d.risk}</span>
                  </div>
                ))}

                {!result.detections.length && <p className="muted">No relevant objects detected.</p>}
              </div>
            </div>
          </section>
        )}

        <div className="info">
          <Info size={18} />
          <span>
            <b>Dataset basis:</b> BDD100K + RWVC-BDD100K. Trained model is preferred when available;
            otherwise the app clearly operates in fallback demo mode for testing and validation.
          </span>
        </div>
      </main>

      <footer>Mine Vehicle Safety AI • React + FastAPI + YOLO + ML Visibility Pipeline</footer>
    </div>
  );
}
