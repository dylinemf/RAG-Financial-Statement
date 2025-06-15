import React, { useRef, useState } from 'react';
import styles from '../styles/FileUpload.module.css';

interface FileUploadProps {
  onUploadComplete?: (result: any) => void;
  onUploadError?: (error: string) => void;
  onProcessingDone?: () => void;
}

// Spinner animasi SVG
const Spinner = () => (
  <span className={styles.spinner}>
    <svg viewBox="0 0 38 38">
      <circle fill="none" stroke="#38b6b8" strokeWidth="4"
        strokeDasharray="55, 150" cx="19" cy="19" r="13"
        style={{ opacity: 0.7 }} />
    </svg>
  </span>
);

export default function FileUpload({
  onUploadComplete,
  onUploadError,
  onProcessingDone,
}: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingDone, setProcessingDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // ==== HANDLER ====
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!f.name.endsWith('.pdf')) {
      setError('File harus PDF!');
      setFile(null);
      return;
    }
    setFile(f);
    setUploadProgress(0);
    setError(null);
    setProcessingDone(false);
  };

  const handleDragOver = (e: React.DragEvent) => e.preventDefault();
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    if (!f.name.endsWith('.pdf')) {
      setError('File harus PDF!');
      setFile(null);
      return;
    }
    setFile(f);
    setUploadProgress(0);
    setError(null);
    setProcessingDone(false);
  };

  // Pooling untuk status proses chunk
  const pollProcessingStatus = () => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('/api/chunks');
        const data = await res.json();
        if (data.total_count && data.total_count > 0) {
          setIsProcessing(false);
          setProcessingDone(true);
          clearInterval(interval);
          if (typeof onProcessingDone === 'function') onProcessingDone();
        }
      } catch {
        // do nothing, poll lagi di interval berikut
      }
    }, 2000);
  };

  const handleUpload = () => {
    if (!file) return;
    setIsUploading(true);
    setUploadProgress(0);
    setError(null);
    setIsProcessing(false);
    setProcessingDone(false);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload');
    xhr.timeout = 15 * 60 * 1000;

    xhr.upload.onprogress = (event: ProgressEvent) => {
      if (event.lengthComputable) {
        setUploadProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      setIsUploading(false);
      setUploadProgress(100);
      try {
        const resp = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) {
          setError(null);
          onUploadComplete && onUploadComplete(resp);

          setIsProcessing(true);
          setProcessingDone(false);
          pollProcessingStatus();
        } else {
          setError(resp.detail || 'Upload error');
          setIsProcessing(false);
          setProcessingDone(false);
          onUploadError && onUploadError(resp.detail || 'Upload error');
        }
      } catch {
        setError('Invalid server response');
        setIsProcessing(false);
        setProcessingDone(false);
        onUploadError && onUploadError('Invalid server response');
      }
    };

    xhr.onerror = () => {
      setIsUploading(false);
      setError('Upload failed');
      setIsProcessing(false);
      setProcessingDone(false);
      onUploadError && onUploadError('Upload failed');
    };

    const form = new FormData();
    form.append('file', file);
    xhr.send(form);
  };

  // ==== RENDER ====
  return (
    <div className={styles.container}>
      {/* AREA DRAG DAN DROP */}
      <div
        className={`${styles.uploadArea}${file ? ` ${styles.uploadAreaHasFile}` : ''}`}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        tabIndex={0}
        role="button"
        title="Klik atau drop file PDF"
      >
        {file
          ? <span>📄 <b>{file.name}</b></span>
          : <>Drag &amp; drop PDF file in here or <u>browse</u></>}
      </div>

      {/* FILE INPUT */}
      <input
        type="file"
        accept=".pdf"
        onChange={handleFileSelect}
        style={{ display: 'none' }}
        ref={inputRef}
      />

      {/* TOMBOL UPLOAD */}
      <button
        type="button"
        className={styles.uploadBtn}
        onClick={handleUpload}
        disabled={!file || isUploading}
      >
        {isUploading ? <><Spinner /> Uploading...</> : 'Upload PDF'}
      </button>

      {/* NOTIFIKASI */}
      {error && <div className={styles.error}>{error}</div>}
      {isUploading && (
        <div className={styles.progressbar}>
          <div className={styles.progress} style={{ width: `${uploadProgress}%` }}>
            {uploadProgress}%
          </div>
        </div>
      )}
      {isProcessing && (
        <div className={styles.info}>
          <Spinner /> PDF is being proccessed on <b>server</b>...
        </div>
      )}
      {processingDone && (
        <div className={styles.success}>
          ✅ Document has been processed! Feel free to start asking the model/Q&amp;A.
        </div>
      )}
    </div>
  );
}