(function () {
  "use strict";

  var CLIPS_META = [
    {
      id: "pen_spin_20261102_141627",
      title: "Pen Spinning",
      fams: ["sharpa", "wuji"],
      defaultFam: "sharpa",
      mvVideo: "./static/videos/retarget/pen_spin_20261102_141627_mv.mp4",
      mvPoster: "./static/thumbs/pen_spin_20261102_141627_mv.jpg"
    },
    {
      id: "rubik_cube_20261028_193815",
      title: "Rubik's Cube",
      fams: ["sharpa", "wuji"],
      defaultFam: "sharpa",
      mvVideo: "./static/videos/retarget/rubik_cube_20261028_193815_mv.mp4",
      mvPoster: "./static/thumbs/rubik_cube_20261028_193815_mv.jpg"
    },
    {
      id: "person_000_object_001_d3",
      title: "Apple",
      fams: ["sharpa", "wuji"],
      defaultFam: "sharpa",
      mvVideo: "./static/videos/retarget/person_000_object_001_d3_mv.mp4",
      mvPoster: "./static/thumbs/person_000_object_001_d3_mv.jpg"
    }
  ];

  var FAM_NAMES = {
    sharpa: "Sharpa",
    wuji: "Wuji"
  };

  function b64ToBuf(s) {
    var bin = atob(s);
    var len = bin.length;
    var u8 = new Uint8Array(len);
    for (var i = 0; i < len; i++) u8[i] = bin.charCodeAt(i);
    return u8.buffer;
  }

  function lin(c255) {
    return Math.pow(c255 / 255.0, 2.2);
  }

  function linColor(r, g, b) {
    return new THREE.Color(lin(r), lin(g), lin(b));
  }

  function buildGeometry(pack) {
    var vi16 = new Int16Array(b64ToBuf(pack.v));
    var fu16 = new Uint16Array(b64ToBuf(pack.f));
    var pos = new Float32Array(vi16.length);
    var s = pack.s;
    for (var i = 0; i < vi16.length; i++) pos[i] = vi16[i] * s;
    var g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setIndex(new THREE.BufferAttribute(fu16, 1));
    if (pack.c) {
      var cu8 = new Uint8Array(b64ToBuf(pack.c));
      var col = new Float32Array(cu8.length);
      for (var j = 0; j < cu8.length; j++) col[j] = lin(cu8[j]);
      g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    }
    g.computeVertexNormals();
    return g;
  }

  function unpackManoClip(mp) {
    return {
      k: mp.k,
      mu: new Int16Array(b64ToBuf(mp.mu)),
      ms: mp.ms,
      sc: new Int16Array(b64ToBuf(mp.sc)),
      ss: mp.ss,
      vt: new Int16Array(b64ToBuf(mp.vt)),
      vs: mp.vs,
      muF: null,
      vtF: null
    };
  }

  function ensureManoFloat(mc) {
    if (mc.muF) return;
    var D = mc.mu.length;
    mc.muF = new Float32Array(D);
    for (var i = 0; i < D; i++) mc.muF[i] = mc.mu[i] * mc.ms;
    mc.vtF = new Float32Array(mc.vt.length);
    for (var j = 0; j < mc.vt.length; j++) mc.vtF[j] = mc.vt[j] * mc.vs;
  }

  // Endless warm-grey alternating checkerboard plane matching the teaser videos' floor squares
  function makeCheckerPlaneTexture() {
    var cv = document.createElement("canvas");
    cv.width = 256;
    cv.height = 256;
    var cx = cv.getContext("2d");
    cx.fillStyle = "rgb(112, 104, 100)";
    cx.fillRect(0, 0, 256, 256);
    cx.fillStyle = "rgb(96, 89, 86)";
    cx.fillRect(0, 0, 128, 128);
    cx.fillRect(128, 128, 128, 128);
    var tex = new THREE.CanvasTexture(cv);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(44, 44); // 16m / 44 -> ~0.18m squares right under the hands & object
    tex.encoding = THREE.sRGBEncoding;
    tex.anisotropy = 4;
    return tex;
  }

  function initRetargetViewer() {
    var rootEl = document.getElementById("rt-viewer-root");
    if (!rootEl || !window.THREE || !window.RETARGET_ASSETS) return;

    var A = window.RETARGET_ASSETS;
    var mvVideo = document.getElementById("rt-mv-video");
    var canvasWrap = document.getElementById("rt-canvas-wrap");
    var loadingEl = document.getElementById("rt-loading");
    var leftTag = document.getElementById("rt-tag-left");
    var rightTag = document.getElementById("rt-tag-right");
    var dividerEl = document.getElementById("rt-split-line");
    var seqBar = document.getElementById("rt-seq-pills");
    var famBar = document.getElementById("rt-fam-pills");
    var modeBar = document.getElementById("rt-mode-pills");
    var playBtn = document.getElementById("rt-play");
    var slider = document.getElementById("rt-slider");
    var timeLabel = document.getElementById("rt-time");
    var speedSel = document.getElementById("rt-speed");
    var resetBtn = document.getElementById("rt-reset-cam");

    var renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    canvasWrap.insertBefore(renderer.domElement, canvasWrap.firstChild);

    var BG_HEX = 0x090809;
    var sceneHuman = new THREE.Scene();
    var sceneRobot = new THREE.Scene();
    sceneHuman.background = new THREE.Color(BG_HEX);
    sceneRobot.background = new THREE.Color(BG_HEX);
    sceneHuman.fog = new THREE.Fog(BG_HEX, 1.15, 3.35);
    sceneRobot.fog = new THREE.Fog(BG_HEX, 1.15, 3.35);

    var sharedCheckerTex = makeCheckerPlaneTexture();
    var sharedPlaneMat = new THREE.MeshStandardMaterial({
      map: sharedCheckerTex,
      color: new THREE.Color(0.36, 0.36, 0.36),
      emissiveMap: sharedCheckerTex,
      emissive: new THREE.Color(0.56, 0.56, 0.56),
      roughness: 1.0,
      metalness: 0.0
    });

    // Camera-mounted 3-point Raymond lights + soft overhead shadow light on the z = 0 checkerboard plane
    var raymondDirsCam = [0, 1, 2].map(function (idx) {
      var th = Math.PI / 6.0;
      var ph = (2.0 * Math.PI * idx) / 3.0;
      return new THREE.Vector3(Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th));
    });

    function setupStage(sc) {
      var amb = new THREE.AmbientLight(0xffffff, 0.28);
      sc.add(amb);

      var topShadow = new THREE.DirectionalLight(0xffffff, 0.42);
      topShadow.position.set(0.22, 0.35, 1.35);
      topShadow.castShadow = true;
      topShadow.shadow.mapSize.set(1024, 1024);
      topShadow.shadow.bias = -0.0004;
      topShadow.shadow.normalBias = 0.002;
      topShadow.shadow.camera.left = -0.55;
      topShadow.shadow.camera.right = 0.55;
      topShadow.shadow.camera.top = 0.55;
      topShadow.shadow.camera.bottom = -0.55;
      topShadow.shadow.camera.near = 0.3;
      topShadow.shadow.camera.far = 2.6;
      sc.add(topShadow);
      sc.add(topShadow.target);

      var rLights = raymondDirsCam.map(function () {
        var dl = new THREE.DirectionalLight(0xffffff, 0.50);
        sc.add(dl);
        sc.add(dl.target);
        return dl;
      });

      // Single endless checkerboard plane directly at z = 0 (contact surface, no wooden slab or legs)
      var groundPlane = new THREE.Mesh(new THREE.PlaneGeometry(16, 16), sharedPlaneMat);
      groundPlane.position.z = -0.0004;
      groundPlane.receiveShadow = true;
      sc.add(groundPlane);

      return rLights;
    }

    var lightsHuman = setupStage(sceneHuman);
    var lightsRobot = setupStage(sceneRobot);

    var camera = new THREE.PerspectiveCamera(35, 1, 0.02, 15);
    camera.up.set(0, 0, 1);
    var controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    controls.minDistance = 0.18;
    controls.maxDistance = 1.85;
    controls.maxPolarAngle = Math.PI * 0.85;

    // Previous warm gold / soft lavender Human hand colors (#ffe592 / #cfa0e6 palette)
    var MATS = {
      mano_left: new THREE.MeshStandardMaterial({ color: linColor(246, 212, 118), roughness: 0.56, metalness: 0.02 }),
      mano_right: new THREE.MeshStandardMaterial({ color: linColor(198, 148, 228), roughness: 0.56, metalness: 0.02 }),
      ghost_left: new THREE.MeshStandardMaterial({ color: linColor(246, 212, 118), roughness: 0.60, transparent: true, opacity: 0.30, depthWrite: false }),
      ghost_right: new THREE.MeshStandardMaterial({ color: linColor(198, 148, 228), roughness: 0.60, transparent: true, opacity: 0.30, depthWrite: false }),
      sharpa_body: new THREE.MeshStandardMaterial({ color: linColor(206, 216, 238), roughness: 0.38, metalness: 0.22 }),
      sharpa_pad: new THREE.MeshStandardMaterial({ color: linColor(42, 184, 110), roughness: 0.52, metalness: 0.05 }),
      wuji_body: new THREE.MeshStandardMaterial({ color: linColor(202, 208, 218), roughness: 0.36, metalness: 0.26 }),
      vc_obj: new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.38, metalness: 0.02 }),
      apple: new THREE.MeshStandardMaterial({ color: linColor(226, 62, 56), roughness: 0.38, metalness: 0.03 })
    };

    var manoHuman = {}, manoGhost = {};
    ["left", "right"].forEach(function (s) {
      var faces = new Uint16Array(b64ToBuf(A.mano_faces[s]));
      var gH = new THREE.BufferGeometry();
      gH.setAttribute("position", new THREE.BufferAttribute(new Float32Array(778 * 3), 3));
      gH.setIndex(new THREE.BufferAttribute(faces, 1));
      var mH = new THREE.Mesh(gH, MATS["mano_" + s]);
      mH.castShadow = true; mH.receiveShadow = true; mH.frustumCulled = false;
      sceneHuman.add(mH);
      manoHuman[s] = mH;

      var mG = new THREE.Mesh(gH, MATS["ghost_" + s]);
      mG.frustumCulled = false;
      mG.visible = false;
      sceneRobot.add(mG);
      manoGhost[s] = mG;
    });

    var robotInstances = {};
    Object.keys(A.robots).forEach(function (fam) {
      var rdef = A.robots[fam];
      var geoms = rdef.meshes.map(buildGeometry);
      var rootGrp = new THREE.Group();
      rootGrp.visible = false;
      var bodyGrps = [];
      for (var i = 0; i < rdef.bodies.length; i++) {
        var bd = rdef.bodies[i];
        var bg = new THREE.Group();
        bg.position.set(bd.pos[0], bd.pos[1], bd.pos[2]);
        bg.quaternion.set(bd.quat[0], bd.quat[1], bd.quat[2], bd.quat[3]);
        bg.userData = {
          jt: bd.jt,
          ja: bd.ja,
          restQuat: new THREE.Quaternion(bd.quat[0], bd.quat[1], bd.quat[2], bd.quat[3]),
          axis: bd.ax ? new THREE.Vector3(bd.ax[0], bd.ax[1], bd.ax[2]) : null
        };
        for (var k = 0; k < bd.g.length; k++) {
          var gd = bd.g[k];
          var mesh = new THREE.Mesh(geoms[gd.m], MATS[gd.mat] || MATS.wuji_body);
          mesh.position.set(gd.p[0], gd.p[1], gd.p[2]);
          mesh.quaternion.set(gd.q[0], gd.q[1], gd.q[2], gd.q[3]);
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          bg.add(mesh);
        }
        if (bd.p === -1) rootGrp.add(bg);
        else bodyGrps[bd.p].add(bg);
        bodyGrps.push(bg);
      }
      sceneRobot.add(rootGrp);
      robotInstances[fam] = { root: rootGrp, bodies: bodyGrps };
    });

    function buildObjectBank(sc) {
      var bank = {};
      Object.keys(A.objects).forEach(function (oname) {
        var odef = A.objects[oname];
        var grp = new THREE.Group();
        grp.visible = false;
        var parts = [];
        if (odef.type === "multi") {
          for (var i = 0; i < odef.parts.length; i++) {
            var gm = buildGeometry(odef.parts[i]);
            var pm = new THREE.Mesh(gm, MATS.vc_obj);
            pm.castShadow = true; pm.receiveShadow = true;
            grp.add(pm);
            parts.push(pm);
          }
        } else {
          var g1 = buildGeometry(odef.mesh);
          var mat = odef.type === "single_vc" ? MATS.vc_obj : (MATS[oname] || MATS.apple);
          var m1 = new THREE.Mesh(g1, mat);
          m1.castShadow = true; m1.receiveShadow = true;
          grp.add(m1);
          parts.push(m1);
        }
        sc.add(grp);
        bank[oname] = { root: grp, parts: parts };
      });
      return bank;
    }

    var objHuman = buildObjectBank(sceneHuman);
    var objRobot = buildObjectBank(sceneRobot);

    var curSeq = CLIPS_META[0].id;
    var curFam = CLIPS_META[0].defaultFam;
    var viewMode = "split";
    var parsedClips = {};
    var activeData = null;
    var frame = 0, playTime = 0, playing = true, speed = 1.0;
    var inViewport = true;
    var tmpQuat = new THREE.Quaternion();
    var tmpVec = new THREE.Vector3();

    function parseClip(seqId) {
      if (parsedClips[seqId]) return parsedClips[seqId];
      var raw = window.RETARGET_CLIPS && window.RETARGET_CLIPS[seqId];
      if (!raw) return null;
      var robotsUnpacked = {};
      Object.keys(raw.robots).forEach(function (f) {
        var r = raw.robots[f];
        robotsUnpacked[f] = {
          w: r.w,
          q: new Int16Array(b64ToBuf(r.q)),
          s: r.s
        };
      });
      var parsed = {
        seq: raw.seq,
        n: raw.n,
        fps: raw.fps,
        objName: raw.obj_name,
        objParts: raw.obj_parts,
        objPos: new Int16Array(b64ToBuf(raw.obj_pos)),
        objPs: raw.obj_ps,
        objQuat: new Int16Array(b64ToBuf(raw.obj_quat)),
        mano: {
          left: unpackManoClip(raw.mano.left),
          right: unpackManoClip(raw.mano.right)
        },
        robots: robotsUnpacked,
        camTgt: raw.cam_tgt,
        camOff: raw.cam_off
      };
      ensureManoFloat(parsed.mano.left);
      ensureManoFloat(parsed.mano.right);
      parsedClips[seqId] = parsed;
      return parsed;
    }

    function loadClipScript(seqId, cb) {
      if (window.RETARGET_CLIPS && window.RETARGET_CLIPS[seqId]) {
        cb(parseClip(seqId));
        return;
      }
      loadingEl.style.display = "flex";
      var s = document.createElement("script");
      s.src = "./static/retarget/clips/" + seqId + ".js";
      s.onload = function () {
        loadingEl.style.display = "none";
        cb(parseClip(seqId));
      };
      s.onerror = function () {
        loadingEl.textContent = "Failed to load clip " + seqId;
      };
      document.head.appendChild(s);
    }

    function syncMvVideo(forceSeek) {
      if (!mvVideo || !activeData) return;
      var want = (frame + 0.2) / activeData.fps;
      if (mvVideo.duration && isFinite(mvVideo.duration)) {
        want = Math.min(want, Math.max(0, mvVideo.duration - 0.01));
      }
      if (playing && inViewport) {
        if (mvVideo.playbackRate !== speed) mvVideo.playbackRate = speed;
        if (forceSeek && mvVideo.readyState >= 1) {
          try { mvVideo.currentTime = want; } catch (e) {}
        }
        if (mvVideo.paused) {
          var p = mvVideo.play();
          if (p && typeof p.catch === "function") p.catch(function () {});
        }
      } else {
        if (!mvVideo.paused) mvVideo.pause();
        if (forceSeek && mvVideo.readyState >= 1) {
          try { mvVideo.currentTime = want; } catch (e2) {}
        }
      }
    }

    function applyFrame(f) {
      if (!activeData) return;
      var C = activeData;

      ["left", "right"].forEach(function (side) {
        var mc = C.mano[side];
        var geom = manoHuman[side].geometry;
        var P = geom.attributes.position.array;
        var K = mc.k, D = P.length;
        var muF = mc.muF, vtF = mc.vtF, sc = mc.sc, ss = mc.ss;
        for (var d = 0; d < D; d++) P[d] = muF[d];
        var scBase = f * K;
        for (var k = 0; k < K; k++) {
          var w = sc[scBase + k] * ss;
          var vtBase = k * D;
          for (var d2 = 0; d2 < D; d2++) {
            P[d2] += w * vtF[vtBase + d2];
          }
        }
        geom.attributes.position.needsUpdate = true;
        geom.computeVertexNormals();
      });

      var obH = objHuman[C.objName].parts;
      var obR = objRobot[C.objName].parts;
      var P_arr = C.objPos, ps = C.objPs, Q_arr = C.objQuat;
      var nParts = C.objParts;
      for (var p = 0; p < nParts; p++) {
        var po = (f * nParts + p) * 3;
        var qo = (f * nParts + p) * 4;
        var px = P_arr[po] * ps, py = P_arr[po + 1] * ps, pz = P_arr[po + 2] * ps;
        var qx = Q_arr[qo] / 32767.0, qy = Q_arr[qo + 1] / 32767.0, qz = Q_arr[qo + 2] / 32767.0, qw = Q_arr[qo + 3] / 32767.0;
        obH[p].position.set(px, py, pz);
        obH[p].quaternion.set(qx, qy, qz, qw);
        obR[p].position.set(px, py, pz);
        obR[p].quaternion.set(qx, qy, qz, qw);
      }

      var rTrack = C.robots[curFam];
      var rInst = robotInstances[curFam];
      if (rTrack && rInst) {
        var Wq = rTrack.w, rq = rTrack.q, rqs = rTrack.s;
        var base = f * Wq;
        var bodies = rInst.bodies;
        for (var b = 0; b < bodies.length; b++) {
          var bg = bodies[b];
          var ud = bg.userData;
          if (ud.jt === "free") {
            var o = base + ud.ja;
            bg.position.set(rq[o] * rqs, rq[o + 1] * rqs, rq[o + 2] * rqs);
            bg.quaternion.set(rq[o + 3] * rqs, rq[o + 4] * rqs, rq[o + 5] * rqs, rq[o + 6] * rqs).normalize();
          } else if (ud.jt === "hinge") {
            var ang = rq[base + ud.ja] * rqs;
            tmpQuat.setFromAxisAngle(ud.axis, ang);
            bg.quaternion.multiplyQuaternions(ud.restQuat, tmpQuat);
          }
        }
      }
    }

    function updateVisibilityAndLabels() {
      Object.keys(robotInstances).forEach(function (f) {
        robotInstances[f].root.visible = (f === curFam);
      });
      Object.keys(objHuman).forEach(function (o) {
        var vis = activeData && (o === activeData.objName);
        objHuman[o].root.visible = vis;
        objRobot[o].root.visible = vis;
      });
      var showGhost = (viewMode === "overlay");
      manoGhost.left.visible = showGhost;
      manoGhost.right.visible = showGhost;

      if (viewMode === "split") {
        leftTag.style.display = "";
        leftTag.textContent = "Human (Ours)";
        rightTag.textContent = FAM_NAMES[curFam] || curFam;
        dividerEl.style.display = "";
      } else if (viewMode === "overlay") {
        leftTag.style.display = "none";
        rightTag.textContent = (FAM_NAMES[curFam] || curFam) + " + Human Overlay";
        dividerEl.style.display = "none";
      } else {
        leftTag.style.display = "none";
        rightTag.textContent = FAM_NAMES[curFam] || curFam;
        dividerEl.style.display = "none";
      }
    }

    function resetCameraForClip() {
      if (!activeData) return;
      var t = activeData.camTgt, o = activeData.camOff;
      controls.target.set(t[0], t[1], t[2]);
      camera.position.set(t[0] + o[0], t[1] + o[1], t[2] + o[2]);
      controls.update();
    }

    function updateUI() {
      if (!activeData) return;
      slider.max = activeData.n - 1;
      slider.value = frame;
      var curS = (frame / activeData.fps).toFixed(1);
      var totS = ((activeData.n - 1) / activeData.fps).toFixed(1);
      timeLabel.textContent = curS + "s / " + totS + "s";
      playBtn.textContent = playing ? "❚❚" : "▶";
    }

    function selectSequence(seqId) {
      var meta = CLIPS_META.filter(function (m) { return m.id === seqId; })[0];
      if (!meta) return;
      curSeq = seqId;
      if (meta.fams.indexOf(curFam) === -1) {
        curFam = meta.defaultFam;
      }
      renderPills();

      if (mvVideo) {
        mvVideo.poster = meta.mvPoster;
        if (mvVideo.getAttribute("src") !== meta.mvVideo) {
          mvVideo.setAttribute("src", meta.mvVideo);
          mvVideo.load();
        }
      }

      loadClipScript(seqId, function (data) {
        activeData = data;
        frame = 0;
        playTime = 0;
        updateVisibilityAndLabels();
        resetCameraForClip();
        applyFrame(0);
        updateUI();
        syncMvVideo(true);
      });
    }

    function renderPills() {
      var meta = CLIPS_META.filter(function (m) { return m.id === curSeq; })[0];
      Array.prototype.forEach.call(seqBar.querySelectorAll("button"), function (b) {
        b.classList.toggle("active", b.dataset.seq === curSeq);
      });
      Array.prototype.forEach.call(famBar.querySelectorAll("button"), function (b) {
        var f = b.dataset.fam;
        b.disabled = meta.fams.indexOf(f) === -1;
        b.classList.toggle("active", f === curFam);
      });
      Array.prototype.forEach.call(modeBar.querySelectorAll("button"), function (b) {
        b.classList.toggle("active", b.dataset.mode === viewMode);
      });
    }

    CLIPS_META.forEach(function (m) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "rt-pill";
      btn.dataset.seq = m.id;
      btn.textContent = m.title;
      btn.onclick = function () { selectSequence(m.id); };
      seqBar.appendChild(btn);
    });

    ["sharpa", "wuji"].forEach(function (f) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "rt-pill";
      btn.dataset.fam = f;
      btn.textContent = FAM_NAMES[f];
      btn.onclick = function () {
        if (btn.disabled) return;
        curFam = f;
        renderPills();
        updateVisibilityAndLabels();
        applyFrame(frame);
      };
      famBar.appendChild(btn);
    });

    [
      { id: "split", label: "Side-by-Side" },
      { id: "overlay", label: "Overlay" },
      { id: "robot", label: "Robot Only" }
    ].forEach(function (m) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "rt-pill subtle";
      btn.dataset.mode = m.id;
      btn.textContent = m.label;
      btn.onclick = function () {
        viewMode = m.id;
        renderPills();
        updateVisibilityAndLabels();
      };
      modeBar.appendChild(btn);
    });

    if (mvVideo) {
      mvVideo.addEventListener("loadeddata", function () {
        syncMvVideo(true);
      });
    }

    playBtn.onclick = function () {
      playing = !playing;
      updateUI();
      syncMvVideo(false);
    };

    slider.oninput = function () {
      if (!activeData) return;
      frame = parseInt(slider.value, 10) || 0;
      playTime = frame / activeData.fps;
      applyFrame(frame);
      updateUI();
      syncMvVideo(true);
    };

    speedSel.onchange = function () {
      speed = parseFloat(speedSel.value) || 1.0;
      syncMvVideo(false);
    };

    resetBtn.onclick = function () {
      resetCameraForClip();
    };

    var canvasW = 800, canvasH = 440;
    function onResize() {
      var rect = canvasWrap.getBoundingClientRect();
      canvasW = Math.max(300, Math.round(rect.width));
      canvasH = Math.max(240, Math.round(rect.height));
      renderer.setSize(canvasW, canvasH, false);
    }
    window.addEventListener("resize", onResize);
    onResize();

    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          inViewport = e.isIntersecting;
          syncMvVideo(false);
        });
      }, { rootMargin: "200px 0px 200px 0px", threshold: 0.01 });
      io.observe(rootEl);
    }

    function updateRaymondLights() {
      camera.updateMatrixWorld();
      for (var i = 0; i < 3; i++) {
        tmpVec.copy(raymondDirsCam[i]).transformDirection(camera.matrixWorld).multiplyScalar(2.0).add(controls.target);
        lightsHuman[i].position.copy(tmpVec);
        lightsHuman[i].target.position.copy(controls.target);
        lightsRobot[i].position.copy(tmpVec);
        lightsRobot[i].target.position.copy(controls.target);
      }
    }

    var lastNow = performance.now();
    function animate(now) {
      requestAnimationFrame(animate);
      var dt = Math.min(0.1, (now - lastNow) / 1000.0);
      lastNow = now;
      if (!inViewport || !activeData) return;

      if (playing) {
        var nf;
        if (mvVideo && !mvVideo.paused && mvVideo.readyState >= 2) {
          playTime = mvVideo.currentTime;
          nf = Math.max(0, Math.min(activeData.n - 1, Math.floor(playTime * activeData.fps)));
        } else {
          playTime += dt * speed;
          var dur = activeData.n / activeData.fps;
          if (playTime >= dur) playTime = playTime % dur;
          nf = Math.min(activeData.n - 1, Math.floor(playTime * activeData.fps));
        }
        if (nf !== frame) {
          frame = nf;
          applyFrame(frame);
          updateUI();
        }
      }

      controls.update();
      updateRaymondLights();

      if (viewMode === "split") {
        var halfW = Math.floor(canvasW / 2);
        var rightW = canvasW - halfW;
        renderer.setScissorTest(true);

        camera.aspect = halfW / canvasH;
        camera.updateProjectionMatrix();
        renderer.setViewport(0, 0, halfW, canvasH);
        renderer.setScissor(0, 0, halfW, canvasH);
        renderer.render(sceneHuman, camera);

        camera.aspect = rightW / canvasH;
        camera.updateProjectionMatrix();
        renderer.setViewport(halfW, 0, rightW, canvasH);
        renderer.setScissor(halfW, 0, rightW, canvasH);
        renderer.render(sceneRobot, camera);
      } else {
        renderer.setScissorTest(false);
        camera.aspect = canvasW / canvasH;
        camera.updateProjectionMatrix();
        renderer.setViewport(0, 0, canvasW, canvasH);
        renderer.render(sceneRobot, camera);
      }
    }

    selectSequence(curSeq);
    requestAnimationFrame(animate);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initRetargetViewer);
  } else {
    initRetargetViewer();
  }
})();
