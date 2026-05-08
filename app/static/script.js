(function () {
  'use strict';

  const $ = (sel) => document.querySelector(sel);

  /* ── DOM refs ──────────────────────────────── */
  const uploadZone    = $('#uploadZone');
  const fileInput     = $('#fileInput');
  const fileName      = $('#fileName');
  const progressWrap  = $('#progressWrap');
  const progressFill  = $('#progressFill');
  const progressText  = $('#progressText');
  const streamPanel   = $('#streamPanel');
  const streamText    = $('#streamText');
  const liveDot       = $('#liveDot');
  const liveLabel     = $('#liveLabel');
  const liveChars     = $('#liveChars');
  const phaseBadge    = $('#phaseBadge');
  const resultWrap    = $('#resultWrap');
  const resultTitle   = $('#resultTitle');
  const resultMeta    = $('#resultMeta');
  const qaContainer   = $('#qaContainer');
  const rawTranscript = $('#rawTranscript');
  const toggleRawBtn  = $('#toggleRawBtn');
  const saveBtn       = $('#saveBtn');
  const copyBtn       = $('#copyBtn');
  const stepUpload    = $('#stepUpload');
  const stepASR       = $('#stepASR');
  const stepQA        = $('#stepQA');

  let currentMarkdown = '';
  let currentFileName = '';
  let selectedFile = null;
  let abortController = null;  // 用于取消正在进行的请求

  /* ── File selection ─────────────────────────── */
  uploadZone.addEventListener('click', () => fileInput.click());
  $('#browseLink').addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
  });

  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });
  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
  });
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
  });

  function handleFile(file) {
    const valid = /\.(m4a|mp3|wav|mp4|aac|ogg|flac|webm)$/i;
    if (!valid.test(file.name)) {
      alert('仅支持音频文件（m4a / mp3 / wav / flac 等）');
      return;
    }
    selectedFile = file;
    currentFileName = file.name;
    fileName.textContent = '已选择: ' + file.name;
    uploadZone.classList.add('has-file');
    startStream();
  }

  /* ═══════════════════════════════════════════════
     Streaming upload via SSE (fetch + ReadableStream)
     ═══════════════════════════════════════════════ */
  async function startStream() {
    if (!selectedFile) return;

    // 取消之前的请求
    if (abortController) abortController.abort();
    abortController = new AbortController();

    // Reset UI
    resetUI();
    progressWrap.classList.add('active');
    progressFill.classList.add('indeterminate');
    progressText.textContent = '正在上传音频...';
    streamPanel.classList.add('active');
    streamText.innerHTML = '<span class="cursor-blink"></span>';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const resp = await fetch('/upload/stream', {
        method: 'POST',
        body: formData,
        signal: abortController.signal,
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: 'HTTP ' + resp.status }));
        throw new Error(err.detail || '服务器错误');
      }

      // ── 读取 SSE 流 ──
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';        // 跨 chunk 的文本缓冲
      let totalChars = 0;     // 累计字符数
      let isFirstChunk = true;

      progressFill.classList.remove('indeterminate');
      progressFill.style.width = '5%';
      progressText.textContent = '正在转写...';
      phaseBadge.textContent = '转写中';
      phaseBadge.className = 'phase-badge phase-transcribing';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE 消息以 \n\n 分隔
        const parts = buffer.split('\n\n');
        buffer = parts.pop();  // 最后一段可能不完整，保留到下次

        for (const part of parts) {
          if (!part.trim()) continue;

          const event = parseSSEEvent(part);
          if (!event) continue;

          switch (event.event) {
            case 'phase':
              handlePhaseEvent(event.data);
              break;

            case 'chunk':
              if (isFirstChunk) {
                // 收到第一个 chunk：清除光标占位符
                streamText.innerHTML = '';
                isFirstChunk = false;
              }
              totalChars += event.data.text.length;
              // 追加转写文本
              streamText.textContent += event.data.text;
              // 自动滚动到底部
              streamText.scrollTop = streamText.scrollHeight;
              // 更新进度
              progressFill.style.width = Math.min(10 + totalChars / 10, 85) + '%';
              liveChars.textContent = '已识别 ' + totalChars + ' 字';
              break;

            case 'qa_done':
              // Q&A 提取完成
              progressFill.style.width = '100%';
              progressFill.classList.remove('indeterminate');
              progressText.textContent =
                event.data.qa_count > 0
                  ? '处理完成！共识别 ' + event.data.qa_count + ' 组问答'
                  : '处理完成（未识别到有效内容）';
              liveDot.classList.add('done');
              liveLabel.textContent = '转写完成';
              phaseBadge.textContent = '已完成';
              phaseBadge.className = 'phase-badge phase-done';

              // 移除光标
              const cursor = streamText.querySelector('.cursor-blink');
              if (cursor) cursor.remove();

              // 渲染 Q&A 结果
              currentMarkdown = event.data.markdown || '';
              renderResult(event.data);
              break;

            case 'error':
              throw new Error(event.data.message || '未知错误');
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      progressText.textContent = '处理失败：' + err.message;
      progressFill.classList.add('indeterminate');
      progressFill.style.width = '100%';
      phaseBadge.textContent = '失败';
      phaseBadge.className = 'phase-badge';
      liveDot.classList.add('done');
      liveLabel.textContent = '转写失败';
      const cursor = streamText.querySelector('.cursor-blink');
      if (cursor) cursor.remove();
    }
  }

  /**
   * 处理 phase 事件：更新步骤指示器和进度文案。
   */
  function handlePhaseEvent(data) {
    const stage = data.stage;
    const msg = data.message || '';

    switch (stage) {
      case 'asr_started':
        // 步骤 1 (上传) 完成，步骤 2 (ASR) 进行中
        setStepDone(stepUpload);
        setStepActive(stepASR);
        setStepPending(stepQA);
        progressFill.classList.remove('indeterminate');
        progressFill.style.width = '5%';
        progressText.textContent = msg;
        liveLabel.textContent = msg;
        phaseBadge.textContent = '转写中';
        phaseBadge.className = 'phase-badge phase-transcribing';
        break;

      case 'asr_complete':
        // 步骤 2 (ASR) 完成
        setStepDone(stepASR);
        progressFill.style.width = '65%';
        progressText.textContent = msg;
        liveChars.textContent = msg;
        break;

      case 'qa_started':
        // 步骤 3 (QA) 进行中
        setStepActive(stepQA);
        progressFill.classList.add('indeterminate');
        progressFill.style.width = '70%';
        progressText.textContent = msg;
        liveLabel.textContent = msg;
        phaseBadge.textContent = '提取中';
        phaseBadge.className = 'phase-badge phase-extracting';
        break;

      case 'qa_complete':
        // 步骤 3 (QA) 完成
        setStepDone(stepQA);
        progressFill.classList.remove('indeterminate');
        progressFill.style.width = '100%';
        progressText.textContent = msg;
        liveDot.classList.add('done');
        liveLabel.textContent = '处理完成';
        phaseBadge.textContent = '已完成';
        phaseBadge.className = 'phase-badge phase-done';
        break;
    }
  }

  function setStepActive(el) {
    el.className = 'step active';
  }
  function setStepDone(el) {
    el.className = 'step done';
  }
  function setStepPending(el) {
    el.className = 'step';
  }

  /**
   * 解析单条 SSE 消息。
   * SSE 格式：
   *   event: <type>\n
   *   data: <json>\n
   */
  function parseSSEEvent(raw) {
    let eventType = '';
    let dataStr = '';
    for (const line of raw.split('\n')) {
      if (line.startsWith('event: ')) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        dataStr = line.slice(6).trim();
      }
    }
    if (!eventType || !dataStr) return null;
    try {
      return { event: eventType, data: JSON.parse(dataStr) };
    } catch {
      return null;
    }
  }

  /* ═══════════════════════════════════════════════
     Render Q&A result
     ═══════════════════════════════════════════════ */
  function renderResult(data) {
    if (!data.markdown) {
      qaContainer.innerHTML = '<div class="empty-state">未获取到有效的问答内容</div>';
      rawTranscript.classList.remove('active');
    } else {
      const lines = data.markdown.split('\n');
      const title = lines[0] ? lines[0].replace(/^# /, '') : '访谈记录';
      resultTitle.textContent = title;
      resultMeta.textContent =
        (data.qa_count ? data.qa_count + ' 组问答' : '') +
        (data.file ? '  ·  ' + data.file : '');

      qaContainer.innerHTML = parseQABlocks(lines);
      rawTranscript.textContent = lines.slice(1).join('\n').trim();
      rawTranscript.classList.remove('active');
    }

    resultWrap.classList.add('active');
    resultWrap.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function parseQABlocks(lines) {
    let html = '';
    let currentQ = '';
    let currentA = '';
    let index = 0;

    for (const line of lines) {
      const qMatch = line.match(/^\*\*Q(\d+)：\*\*\s*(.*)/);
      const aMatch = line.match(/^\*\*A(\d+)：\*\*\s*(.*)/);
      if (qMatch) {
        if (currentQ || currentA) { index++; html += buildQABlock(index, currentQ, currentA); }
        currentQ = qMatch[2];
        currentA = '';
      } else if (aMatch) {
        currentA = aMatch[2];
      } else if (line.trim() && !line.startsWith('#')) {
        if (currentA) currentA += '\n' + line;
      }
    }
    if (currentQ || currentA) { index++; html += buildQABlock(index, currentQ, currentA); }
    return html || '<div class="empty-state">未能解析问答内容</div>';
  }

  function buildQABlock(idx, q, a) {
    const escapedQ = escapeHtml(q || '（未识别的问题）');
    const escapedA = escapeHtml(a || '（未识别的回答）').replace(/\n/g, '<br>');
    return `
      <div class="qa-block">
        <div class="q"><span class="qa-index">${idx}</span>${escapedQ}</div>
        <div class="a">${escapedA}</div>
      </div>`;
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  /* ── Reset UI for new upload ─────────────────── */
  function resetUI() {
    resultWrap.classList.remove('active');
    rawTranscript.classList.remove('active');
    toggleRawBtn.textContent = '查看原始转写';
    streamPanel.classList.remove('active');
    streamText.textContent = '';
    liveDot.classList.remove('done');
    liveLabel.textContent = '实时转写中...';
    liveChars.textContent = '';
    phaseBadge.textContent = '转写中';
    phaseBadge.className = 'phase-badge phase-transcribing';
    qaContainer.innerHTML = '';
    rawTranscript.textContent = '';
    progressFill.style.width = '0%';
    progressFill.classList.remove('indeterminate');
    currentMarkdown = '';
    // 重置步骤指示器
    setStepActive(stepUpload);
    setStepPending(stepASR);
    setStepPending(stepQA);
  }

  /* ═══════════════════════════════════════════════
     Action buttons
     ═══════════════════════════════════════════════ */

  // 查看原始转写
  toggleRawBtn.addEventListener('click', () => {
    rawTranscript.classList.toggle('active');
    toggleRawBtn.textContent = rawTranscript.classList.contains('active')
      ? '隐藏原始转写' : '查看原始转写';
  });

  // 保存 Markdown 文件
  saveBtn.addEventListener('click', () => {
    if (!currentMarkdown) {
      alert('暂无内容可保存');
      return;
    }
    const blob = new Blob([currentMarkdown], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    // 文件名：使用原始音频名 + 时间戳
    const ts = new Date().toISOString().slice(0, 19).replace(/T/, '_').replace(/:/g, '');
    const base = currentFileName.replace(/\.[^.]+$/, '') || 'interview';
    a.download = base + '_' + ts + '.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  // 复制内容
  copyBtn.addEventListener('click', async () => {
    if (!currentMarkdown) {
      alert('暂无内容可复制');
      return;
    }
    try {
      await navigator.clipboard.writeText(currentMarkdown);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = currentMarkdown;
      ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    const orig = copyBtn.textContent;
    copyBtn.textContent = '已复制！';
    setTimeout(() => { copyBtn.textContent = orig; }, 2000);
  });

})();
