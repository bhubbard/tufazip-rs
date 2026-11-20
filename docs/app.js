'use strict';

(function () {
  const modeSelect = document.getElementById('cli-mode');
  const backendSelect = document.getElementById('cli-backend');
  const presetSelect = document.getElementById('cli-preset');
  const presetHint = document.getElementById('cli-preset-hint');
  const releaseCheckbox = document.getElementById('cli-release');
  const outputEl = document.getElementById('cli-output');
  const noteEl = document.getElementById('cli-note');
  const copyCliBtn = document.getElementById('copy-cli-btn');
  const liveStatus = document.getElementById('live-status');

  const MODES = {
    compress: {
      usesPreset: true,
      usesDevice: true,
      buildArgs: (preset, device) =>
        `compress input.txt archive.tz --vocab out.voc --preset ${preset}${device}`,
      note:
        'Induces an NNCP dictionary in out.voc, tokenizes input.txt, and writes a neural arithmetic-coded artifact.',
    },
    decompress: {
      usesPreset: true,
      usesDevice: true,
      buildArgs: (preset, device) =>
        `decompress archive.tz recovered.txt --vocab out.voc --preset ${preset}${device}`,
      note:
        'Replays identical Transformer online updates with out.voc to restore recovered.txt byte-for-byte.',
    },
    'codec-compress': {
      usesPreset: true,
      usesDevice: true,
      buildArgs: (preset, device) =>
        `codec --input tokens.pp --output payload.bin --preset ${preset}${device} --checkpoint-dir ./ckpt`,
      note:
        'Compresses a pre-tokenized .pp stream (nn.py equivalent) with resumable safetensors + arithmetic checkpoints.',
    },
    'codec-decompress': {
      usesPreset: true,
      usesDevice: true,
      buildArgs: (preset, device) =>
        `codec --decompress --input payload.bin --output decoded.pp --preset ${preset}${device} --checkpoint-dir ./ckpt`,
      note:
        'Decompresses payload.bin back into big-endian u16 tokens in decoded.pp using resumable checkpoints.',
    },
    'preprocess-encode': {
      usesPreset: false,
      usesDevice: false,
      buildArgs: () => 'preprocess encode out.voc input.txt tokens.pp 4097 5',
      note:
        'Mirrors Bellard preprocess.c: induces up to 4,097 symbols (min bigram freq 5) and writes out.voc + tokens.pp.',
    },
    'preprocess-decode': {
      usesPreset: false,
      usesDevice: false,
      buildArgs: () => 'preprocess decode out.voc tokens.pp recovered.txt',
      note:
        'Streams big-endian tokens from tokens.pp back to exact raw bytes using out.voc.',
    },
    'inspect-prior': {
      usesPreset: false,
      usesDevice: false,
      buildArgs: () => 'inspect-prior assets/weights.bin --vocab assets/enwik9.voc',
      note:
        'Reads the QPI1 archive header, checks vocab SHA-256, and prints parameter and loop-untying layer schedule stats.',
    },
    'v2-decode': {
      usesPreset: false,
      usesDevice: true,
      buildArgs: (_preset, device) =>
        `v2-decode enwik9-v2.bin enwik9 ./tufazip-v2-state${device}`,
      note:
        'Runs the v2 verifier/decoder wrapper against assets/weights.bin and assets/enwik9.voc (add --skip-verify on non-B200 hardware).',
    },
  };

  const BACKENDS = {
    cpu: { feature: '', deviceFlag: ' --device cpu' },
    accelerate: { feature: ' --features tufazip-codec/accelerate', deviceFlag: ' --device cpu' },
    metal: { feature: ' --features tufazip-codec/metal', deviceFlag: ' --device metal' },
    cuda: { feature: ' --features tufazip-codec/cuda', deviceFlag: ' --device cuda' },
    mkl: { feature: ' --features tufazip-codec/mkl', deviceFlag: ' --device cpu' },
  };

  function announce(message) {
    if (!liveStatus) {
      return;
    }
    liveStatus.textContent = '';
    window.setTimeout(() => {
      liveStatus.textContent = message;
    }, 20);
  }

  function updateCommand() {
    if (!modeSelect || !backendSelect || !presetSelect || !releaseCheckbox || !outputEl) {
      return;
    }

    const modeKey = modeSelect.value;
    const backendKey = backendSelect.value;
    const presetKey = presetSelect.value;

    const modeCfg = MODES[modeKey] || MODES.compress;
    const backendCfg = BACKENDS[backendKey] || BACKENDS.cpu;

    presetSelect.disabled = !modeCfg.usesPreset;
    if (presetHint) {
      presetHint.textContent = modeCfg.usesPreset
        ? 'tiny runs fast locally; v2 sets full 12-layer enwik9 geometry.'
        : `Not used by ${modeKey}; subcommand supplies its own fixed configuration.`;
    }

    const releaseFlag = releaseCheckbox.checked ? ' --release' : '';
    const featureFlag = modeCfg.usesDevice ? backendCfg.feature : '';
    const deviceArg = modeCfg.usesDevice ? backendCfg.deviceFlag : '';
    const args = modeCfg.buildArgs(presetKey, deviceArg);

    outputEl.textContent = `cargo run${releaseFlag}${featureFlag} --bin tufazip -- ${args}`;
    if (noteEl) {
      noteEl.textContent = modeCfg.note;
    }
  }

  async function copyText(text, button, labelDescription) {
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
        await navigator.clipboard.writeText(text);
      } else {
        const helper = document.createElement('textarea');
        helper.value = text;
        helper.setAttribute('readonly', '');
        helper.style.position = 'fixed';
        helper.style.opacity = '0';
        document.body.appendChild(helper);
        helper.select();
        document.execCommand('copy');
        document.body.removeChild(helper);
      }

      const hidden = button.querySelector('.visually-hidden');
      const hiddenHtml = hidden ? hidden.outerHTML : '';
      button.innerHTML = `Copied ${hiddenHtml}`;
      announce(`Copied ${labelDescription} to clipboard.`);

      window.setTimeout(() => {
        button.innerHTML = `Copy ${hiddenHtml}`;
      }, 1800);
    } catch (_err) {
      announce(`Could not copy ${labelDescription}; select the code block manually.`);
    }
  }

  if (modeSelect && backendSelect && presetSelect && releaseCheckbox) {
    modeSelect.addEventListener('change', updateCommand);
    backendSelect.addEventListener('change', updateCommand);
    presetSelect.addEventListener('change', updateCommand);
    releaseCheckbox.addEventListener('change', updateCommand);
    updateCommand();
  }

  if (copyCliBtn && outputEl) {
    copyCliBtn.addEventListener('click', () => {
      copyText(outputEl.textContent || '', copyCliBtn, 'generated CLI command');
    });
  }

  document.querySelectorAll('.copy-snippet-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-copy-target');
      const targetEl = targetId ? document.getElementById(targetId) : null;
      if (targetEl) {
        copyText(targetEl.textContent || '', btn, 'command snippet');
      }
    });
  });
})();
