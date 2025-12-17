
  // Mouse elements
  const startMouseBtn = document.getElementById("startMouse");
  const stopMouseBtn = document.getElementById("stopMouse");
  const mouseVideo = document.getElementById("mouseVideo");
  const mousePlaceholder = document.getElementById("mousePlaceholder");
  const mouseCardStatus = document.getElementById("mouseCardStatus");
  const mouseStatusText = document.getElementById("mouseStatusText");
  const mouseOverlay = document.getElementById("mouseOverlay");

  // Posture elements
  const startPostureBtn = document.getElementById("startPosture");
  const stopPostureBtn = document.getElementById("stopPosture");
  const postureVideo = document.getElementById("postureVideo");
  const posturePlaceholder = document.getElementById("posturePlaceholder");
  const postureCardStatus = document.getElementById("postureCardStatus");
  const postureStatusText = document.getElementById("postureStatusText");
  const postureOverlay = document.getElementById("postureOverlay");

  // ================= START MOUSE =================
  startMouseBtn.addEventListener("click", async () => {
    startMouseBtn.disabled = true;

    try {
      const res = await fetch("/start_mouse", { method: "POST" });
      if (res.ok) {
        mouseVideo.src = "/video_mouse?t=" + Date.now();
        mouseVideo.style.display = "block";
        mousePlaceholder.style.display = "none";
        mouseOverlay.style.display = "block";

        // ✅ STATUS UPDATE
        mouseCardStatus.className = "card-status active";
        mouseCardStatus.innerHTML =
          '<span class="pulse-dot"></span><span>Active</span>';

        mouseStatusText.innerHTML =
          'Active <span class="status-badge badge-active">Online</span>';

        stopMouseBtn.disabled = false;
      }
    } catch (e) {
      alert("Error starting mouse control");
      startMouseBtn.disabled = false;
    }
  });

  // ================= STOP MOUSE =================
  stopMouseBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/stop_mouse", { method: "POST" });
      if (res.ok) {
        mouseVideo.style.display = "none";
        mousePlaceholder.style.display = "flex";
        mouseOverlay.style.display = "none";

        // ✅ STATUS UPDATE
        mouseCardStatus.className = "card-status inactive";
        mouseCardStatus.innerHTML =
          '<span class="pulse-dot"></span><span>Inactive</span>';

        mouseStatusText.innerHTML =
          'Inactive <span class="status-badge badge-inactive">Offline</span>';

        startMouseBtn.disabled = false;
        stopMouseBtn.disabled = true;
      }
    } catch (e) {
      alert("Error stopping mouse control");
    }
  });

  // ================= START POSTURE =================
  startPostureBtn.addEventListener("click", async () => {
    startPostureBtn.disabled = true;

    try {
      const res = await fetch("/start_posture", { method: "POST" });
      if (res.ok) {
        postureVideo.src = "/video_posture?t=" + Date.now();
        postureVideo.style.display = "block";
        posturePlaceholder.style.display = "none";
        postureOverlay.style.display = "block";

        // ✅ STATUS UPDATE
        postureCardStatus.className = "card-status active";
        postureCardStatus.innerHTML =
          '<span class="pulse-dot"></span><span>Active</span>';

        postureStatusText.innerHTML =
          'Active <span class="status-badge badge-active">Online</span>';

        stopPostureBtn.disabled = false;
      }
    } catch (e) {
      alert("Error starting posture monitor");
      startPostureBtn.disabled = false;
    }
  });

  // ================= STOP POSTURE =================
  stopPostureBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/stop_posture", { method: "POST" });
      if (res.ok) {
        postureVideo.style.display = "none";
        posturePlaceholder.style.display = "flex";
        postureOverlay.style.display = "none";

        // ✅ STATUS UPDATE
        postureCardStatus.className = "card-status inactive";
        postureCardStatus.innerHTML =
          '<span class="pulse-dot"></span><span>Inactive</span>';

        postureStatusText.innerHTML =
          'Inactive <span class="status-badge badge-inactive">Offline</span>';

        startPostureBtn.disabled = false;
        stopPostureBtn.disabled = true;
      }
    } catch (e) {
      alert("Error stopping posture monitor");
    }
  });


  
