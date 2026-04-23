var API = '';
var workflowId = null;
var seenMessageCount = 0;
var pollTimer = null;

var voiceEnabled = false;
var inVoiceCall = false;
var voiceCallShown = false;

var mediaStream = null;
var mediaRecorder = null;
var audioChunks = [];
var isRecording = false;

var audioCtx = null;
var analyserNode = null;
var vadFrameId = null;
var vadActive = false;
var speechDetected = false;
var silenceStart = 0;

var fillerBlob = null;
var pendingRealAudio = null;
var fillerPlaying = false;
var voiceTurnInFlight = false;
var agentAudioPlaying = false;
var pendingStageAdvance = null;

var SPEECH_THRESHOLD = 30;
var SILENCE_DURATION_MS = 1500;

/* ---- Config ---- */

async function fetchConfig() {
  try {
    var res = await fetch(API + '/config');
    if (res.ok) {
      var cfg = await res.json();
      voiceEnabled = cfg.agent2_voice_enabled === true;
    }
  } catch (_) {}
}

/* ---- Pipeline start ---- */

async function startPipeline() {
  var btn = document.getElementById('btn-start');
  btn.disabled = true;
  btn.textContent = 'Starting...';

  await fetchConfig();

  var body = {
    borrower: {
      borrower_id: document.getElementById('f-bid').value,
      account_reference: document.getElementById('f-acct').value,
      debt_amount: parseFloat(document.getElementById('f-debt').value),
      days_past_due: parseInt(document.getElementById('f-dpd').value),
      borrower_message: document.getElementById('f-msg').value,
    }
  };

  try {
    var res = await fetch(API + '/pipelines', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      var err = await res.json().catch(function() { return {}; });
      throw new Error(err.detail || res.statusText);
    }
    var data = await res.json();
    workflowId = data.workflow_id;

    document.getElementById('setup').style.display = 'none';
    document.getElementById('chat-area').style.display = 'flex';
    document.getElementById('sb-wf').textContent = workflowId;

    addSystemMsg('Pipeline started. Workflow: ' + workflowId);
    if (voiceEnabled) addSystemMsg('Voice mode enabled for Resolution stage.');
    addBorrowerMsg(body.borrower.borrower_message);

    seenMessageCount = 0;
    voiceCallShown = false;
    startPolling();
  } catch (e) {
    addSystemMsg('ERROR: ' + e.message);
    btn.disabled = false;
    btn.textContent = 'Start Pipeline';
  }
}

/* ---- Text messaging ---- */

async function sendMessage() {
  var input = document.getElementById('chat-input');
  var text = input.value.trim();
  if (!text || !workflowId) return;
  input.value = '';

  addBorrowerMsg(text);

  try {
    var res = await fetch(API + '/pipelines/' + workflowId + '/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok) {
      var err = await res.json().catch(function() { return {}; });
      throw new Error(err.detail || res.statusText);
    }
  } catch (e) {
    addSystemMsg('Send failed: ' + e.message);
  }
}

/* ---- Polling ---- */

async function pollStatus() {
  if (!workflowId) return;
  try {
    var res = await fetch(API + '/pipelines/' + workflowId);
    if (!res.ok) return;
    var data = await res.json();

    document.getElementById('sb-stage').textContent = data.current_stage;
    document.getElementById('sb-status').textContent =
      data.completed ? 'completed' : data.failed ? 'failed' : 'running';

    var badge = document.getElementById('stage-badge');
    badge.style.display = 'inline';
    badge.textContent = data.current_stage;
    badge.className = 'badge ' + (data.completed ? 'badge-done' : data.failed ? 'badge-err' : 'badge-stage');

    if (voiceEnabled && data.current_stage === 'resolution' && !voiceCallShown && !inVoiceCall) {
      voiceCallShown = true;
      showIncomingCall();
    }

    if (inVoiceCall && data.current_stage !== 'resolution') {
      pendingStageAdvance = data.current_stage;
      maybeFinishResolutionCall();
    }

    if (!inVoiceCall) {
      var transcript = data.transcript || [];
      while (seenMessageCount < transcript.length) {
        var msg = transcript[seenMessageCount];
        if (msg.role === 'agent') {
          addAgentMsg(msg.text, msg.stage);
        }
        seenMessageCount++;
      }
    }

    if (data.completed || data.failed) {
      stopPolling();
      if (inVoiceCall) endCall();
      var label = data.completed ? 'Pipeline completed' : 'Pipeline failed';
      addSystemMsg(label + (data.final_outcome ? ' (' + data.final_outcome + ')' : '') + (data.error ? ': ' + data.error : ''));
      document.getElementById('chat-input').disabled = true;
    }
  } catch (_) {}
}

function startPolling() {
  pollStatus();
  pollTimer = setInterval(pollStatus, 1500);
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

/* ---- Incoming call card ---- */

function showIncomingCall() {
  document.getElementById('call-incoming').style.display = 'flex';
  document.getElementById('input-bar').style.display = 'none';
}

function hideIncomingCall() {
  document.getElementById('call-incoming').style.display = 'none';
}

async function acceptCall() {
  hideIncomingCall();
  inVoiceCall = true;
  document.getElementById('call-panel').style.display = 'flex';
  document.getElementById('input-bar').style.display = 'none';
  setCallStatus('speaking');
  addSystemMsg('Resolution voice call started.');

  /* Fetch filler audio in background */
  fetchFillerAudio();

  /* Fetch and play greeting */
  try {
    var res = await fetch(API + '/voice-greeting?workflow_id=' + encodeURIComponent(workflowId));
    if (res.ok) {
      var data = await res.json();
      if (data.audio_base64 && data.audio_mime) {
        addAgentMsg(data.assistant_reply, 'resolution');
        seenMessageCount += 1;
        await playAudioAndWait(data.audio_base64, data.audio_mime);
      }
    }
  } catch (_) {
    addSystemMsg('Could not fetch greeting audio.');
  }

  if (inVoiceCall) {
    await initVAD();
    beginListening();
  }
}

function declineCall() {
  hideIncomingCall();
  document.getElementById('input-bar').style.display = 'flex';
  addSystemMsg('Voice call declined. Using text mode for Resolution.');
}

function endCall() {
  inVoiceCall = false;
  vadActive = false;
  voiceTurnInFlight = false;
  agentAudioPlaying = false;
  pendingStageAdvance = null;
  if (vadFrameId) { cancelAnimationFrame(vadFrameId); vadFrameId = null; }
  stopRecordingSilent();
  if (mediaStream) {
    mediaStream.getTracks().forEach(function(t) { t.stop(); });
    mediaStream = null;
  }
  if (audioCtx) {
    audioCtx.close().catch(function() {});
    audioCtx = null;
    analyserNode = null;
  }
  fillerBlob = null;
  pendingRealAudio = null;
  fillerPlaying = false;
  document.getElementById('call-panel').style.display = 'none';
  document.getElementById('input-bar').style.display = 'flex';
  document.getElementById('tts-player').pause();
  document.getElementById('volume-bar').style.width = '0%';
}

function maybeFinishResolutionCall() {
  if (!pendingStageAdvance || !inVoiceCall) return;
  if (voiceTurnInFlight || fillerPlaying || agentAudioPlaying) return;

  var nextStage = pendingStageAdvance;
  pendingStageAdvance = null;
  endCall();
  addSystemMsg('Resolution call ended - stage advanced to ' + nextStage);
}

/* ---- Call status pill ---- */

function setCallStatus(state) {
  var el = document.getElementById('call-status');
  var labels = { idle: 'Idle', listening: 'Listening', thinking: 'Thinking...', speaking: 'Agent Speaking' };
  var pills = { idle: 'pill-idle', listening: 'pill-listening', thinking: 'pill-thinking', speaking: 'pill-speaking' };
  el.textContent = labels[state] || state;
  el.className = 'call-status-pill ' + (pills[state] || 'pill-idle');
}

/* ---- Filler audio ---- */

async function fetchFillerAudio() {
  try {
    var res = await fetch('/static/filler.mp3');
    if (res.ok) {
      fillerBlob = await res.blob();
      console.log('[voice] filler.mp3 cached', fillerBlob.size, 'bytes');
    } else {
      console.warn('[voice] filler.mp3 fetch failed:', res.status);
    }
  } catch (e) { console.warn('[voice] filler.mp3 fetch error:', e); }
}

function playFillerAudio() {
  if (!fillerBlob) { console.warn('[voice] no filler blob cached'); return false; }
  fillerPlaying = true;
  var url = URL.createObjectURL(fillerBlob);
  var player = document.getElementById('tts-player');
  player.onended = null;
  player.pause();
  player.src = url;
  player.onended = function() {
    URL.revokeObjectURL(url);
    fillerPlaying = false;
    onFillerEnded();
  };
  player.play().then(function() {
    console.log('[voice] filler playing');
  }).catch(function(e) {
    console.warn('[voice] filler play() rejected:', e);
    fillerPlaying = false;
    onFillerEnded();
  });
  return true;
}

function onFillerEnded() {
  if (!inVoiceCall) return;
  if (pendingRealAudio) {
    var real = pendingRealAudio;
    pendingRealAudio = null;
    setCallStatus('speaking');
    playRealAudio(real.audio_base64, real.audio_mime);
    return;
  }

  maybeFinishResolutionCall();
}

function playRealAudio(b64, mime) {
  if (!inVoiceCall) return;
  var raw = atob(b64);
  var arr = new Uint8Array(raw.length);
  for (var i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
  var blob = new Blob([arr], { type: mime });
  var url = URL.createObjectURL(blob);

  var player = document.getElementById('tts-player');
  player.src = url;
  agentAudioPlaying = true;
  player.onended = function() {
    URL.revokeObjectURL(url);
    agentAudioPlaying = false;
    if (pendingStageAdvance) {
      maybeFinishResolutionCall();
      return;
    }
    if (inVoiceCall) {
      beginListening();
    }
  };
  setCallStatus('speaking');
  player.play().catch(function() {
    agentAudioPlaying = false;
    if (pendingStageAdvance) {
      maybeFinishResolutionCall();
      return;
    }
    if (inVoiceCall) beginListening();
  });
}

function playAudioAndWait(b64, mime) {
  return new Promise(function(resolve) {
    var raw = atob(b64);
    var arr = new Uint8Array(raw.length);
    for (var i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
    var blob = new Blob([arr], { type: mime });
    var url = URL.createObjectURL(blob);
    var player = document.getElementById('tts-player');
    player.src = url;
    agentAudioPlaying = true;
    player.onended = function() {
      URL.revokeObjectURL(url);
      agentAudioPlaying = false;
      maybeFinishResolutionCall();
      resolve();
    };
    player.play().catch(function() {
      agentAudioPlaying = false;
      maybeFinishResolutionCall();
      resolve();
    });
  });
}

/* ---- VAD (Voice Activity Detection) ---- */

async function initVAD() {
  if (!mediaStream) {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  var source = audioCtx.createMediaStreamSource(mediaStream);
  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 512;
  analyserNode.smoothingTimeConstant = 0.3;
  source.connect(analyserNode);
}

function beginListening() {
  if (!inVoiceCall) return;
  setCallStatus('listening');
  speechDetected = false;
  silenceStart = 0;
  vadActive = true;
  vadLoop();
}

function vadLoop() {
  if (!vadActive || !inVoiceCall) return;

  var bufLen = analyserNode.frequencyBinCount;
  var dataArray = new Uint8Array(bufLen);
  analyserNode.getByteFrequencyData(dataArray);

  var sum = 0;
  for (var i = 0; i < bufLen; i++) sum += dataArray[i];
  var avgVolume = sum / bufLen;

  var pct = Math.min(100, Math.round((avgVolume / 128) * 100));
  document.getElementById('volume-bar').style.width = pct + '%';

  var now = Date.now();

  if (avgVolume >= SPEECH_THRESHOLD) {
    if (!speechDetected) {
      speechDetected = true;
      startVADRecording();
    }
    silenceStart = 0;
  } else if (speechDetected) {
    if (!silenceStart) {
      silenceStart = now;
    } else if (now - silenceStart >= SILENCE_DURATION_MS) {
      vadActive = false;
      stopRecordingAndProcess();
      return;
    }
  }

  vadFrameId = requestAnimationFrame(vadLoop);
}

function startVADRecording() {
  if (isRecording) return;
  audioChunks = [];
  var mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus'
    : 'audio/webm';
  mediaRecorder = new MediaRecorder(mediaStream, { mimeType: mimeType });
  mediaRecorder.ondataavailable = function(e) {
    if (e.data.size > 0) audioChunks.push(e.data);
  };
  mediaRecorder.start();
  isRecording = true;
}

function stopRecordingSilent() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.onstop = null;
    mediaRecorder.stop();
  }
  isRecording = false;
  audioChunks = [];
}

function stopRecordingAndProcess() {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') {
    beginListening();
    return;
  }
  mediaRecorder.onstop = onVADRecordingDone;
  mediaRecorder.stop();
  isRecording = false;
}

async function onVADRecordingDone() {
  var mimeType = (mediaRecorder && mediaRecorder.mimeType) ? mediaRecorder.mimeType : 'audio/webm';
  var blob = new Blob(audioChunks, { type: mimeType });
  audioChunks = [];

  if (blob.size === 0) {
    if (inVoiceCall) beginListening();
    return;
  }

  setCallStatus('thinking');
  document.getElementById('volume-bar').style.width = '0%';
  pendingRealAudio = null;

  var fillerStarted = playFillerAudio();

  var form = new FormData();
  form.append('audio', blob, 'recording.webm');
  voiceTurnInFlight = true;

  try {
    var res = await fetch(API + '/pipelines/' + workflowId + '/voice-turn', {
      method: 'POST',
      body: form,
    });
    if (!res.ok) {
      var err = await res.json().catch(function() { return {}; });
      throw new Error(err.detail || res.statusText);
    }
    var data = await res.json();

    if (!inVoiceCall) {
      return;
    }

    seenMessageCount += 2;
    addBorrowerMsg(data.transcribed_text);
    addAgentMsg(data.assistant_reply, 'resolution');

    if (data.current_stage !== 'resolution' || data.stage_complete) {
      addSystemMsg('Resolution stage complete.');
    }

    if (!data.audio_base64 || !data.audio_mime) {
      maybeFinishResolutionCall();
      if (inVoiceCall) beginListening();
      return;
    }

    if (fillerPlaying) {
      pendingRealAudio = { audio_base64: data.audio_base64, audio_mime: data.audio_mime };
    } else {
      setCallStatus('speaking');
      playRealAudio(data.audio_base64, data.audio_mime);
    }
  } catch (e) {
    addSystemMsg('Voice turn failed: ' + e.message);
    if (inVoiceCall) beginListening();
  } finally {
    voiceTurnInFlight = false;
    maybeFinishResolutionCall();
  }
}

/* ---- Chat message helpers ---- */

function addBorrowerMsg(text) {
  var el = document.createElement('div');
  el.className = 'msg msg-borrower';
  el.textContent = text;
  appendMsg(el);
}

function addAgentMsg(text, stage) {
  var el = document.createElement('div');
  el.className = 'msg msg-agent';
  el.textContent = text;
  if (stage) {
    var s = document.createElement('div');
    s.className = 'msg-stage';
    s.textContent = stage;
    el.appendChild(s);
  }
  appendMsg(el);
}

function addSystemMsg(text) {
  var el = document.createElement('div');
  el.className = 'msg msg-system';
  el.textContent = text;
  appendMsg(el);
}

function appendMsg(el) {
  var container = document.getElementById('messages');
  container.appendChild(el);
  container.scrollTop = container.scrollHeight;
}