// script.js

document.addEventListener('DOMContentLoaded', () => {
    // 1. Configure Marked with proper settings for bullet points and other formatting
    marked.setOptions({
        gfm: true,
        breaks: true,
        sanitize: false,
        smartLists: true,
        xhtml: true
    });

    // Elements
    const videoForm = document.getElementById('videoForm');
    const submitBtn = document.getElementById('submitBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const videoDataContainer = document.getElementById('videoData');
    const videoTitle = document.getElementById('videoTitle');
    const videoDescription = document.getElementById('videoDescription');
    const videoTranscript = document.getElementById('videoTranscript');
    const videoSummary = document.getElementById('videoSummary');
    const summaryContent = document.getElementById('summaryContent');
    const playerSection = document.getElementById('playerSection');
    const toggleSidebar = document.getElementById('toggleSidebar');
    const toggleSidebarDesktop = document.getElementById('toggleSidebarDesktop');
    const sidebar = document.getElementById('sidebar');
    const closeSidebar = document.getElementById('closeSidebar');
    const followUpForm = document.getElementById('followUpForm');
    const followUpInput = document.getElementById('followUpInput');
    const followUpSubmit = document.getElementById('followUpSubmit');
    const dragHandle = document.getElementById('dragHandle');
    const rateLimitMessage = document.getElementById('rateLimitMessage');
    const errorContainer = document.getElementById('errorContainer');

    // YouTube Player variables
    let player;
    let currentVideoId = null;

    // New: Add a "Copy Summary" button
    let copySummaryBtn = document.createElement('button');
    copySummaryBtn.textContent = "Copy Summary";
    copySummaryBtn.className = "bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg mt-4";
    copySummaryBtn.addEventListener('click', () => {
        const textToCopy = summaryContent.textContent || "";
        navigator.clipboard.writeText(textToCopy)
            .then(() => showError("Summary copied to clipboard!", 2000))
            .catch(err => showError("Failed to copy summary: " + err.message));
    });

    // State
    let conversationHistory = [];
    let lastMessageTime = 0;
    let rateLimitResetTimeout;
    const rateLimitCooldown = 60000; // 1 minute
    const minSidebarWidth = 300;
    // Increase the max to let the user resize more
    const maxSidebarWidth = 900;
    let isDragging = false;
    let initialX;
    let initialWidth;

    // Utility: parse Markdown with Marked
    const formatResponse = (text) => {
        return marked.parse(text || '');
    };

    // Initialize YouTube Player
    function initYouTubePlayer(videoId) {
        if (!videoId) return;
        
        currentVideoId = videoId;
        
        // If player already exists, just load new video
        if (player) {
            player.loadVideoById(videoId);
            return;
        }
        
        // Create new player
        player = new YT.Player('youtubePlayer', {
            height: '100%',
            width: '100%',
            videoId: videoId,
            playerVars: {
                'playsinline': 1,
                'rel': 0,
                'modestbranding': 1
            },
            events: {
                'onReady': onPlayerReady
            }
        });
    }
    
    function onPlayerReady(event) {
        // Player is ready
        console.log('YouTube player is ready');
        playerSection.classList.remove('hidden');
    }
    
    // Function to seek to specific time in the video
    function seekToTime(seconds) {
        if (player && player.seekTo) {
            player.seekTo(seconds, true);
            player.playVideo();
        }
    }
    
    // Parse timestamp string to seconds
    function parseTimestamp(timestamp) {
        // Handle ~MM:SS format (e.g., "~5:30")
        if (timestamp.startsWith('~')) {
            timestamp = timestamp.substring(1);
        }
        
        // Extract only the time portion if there's additional text
        const timePattern = /(\d+):(\d+)/;
        const match = timestamp.match(timePattern);
        
        if (match) {
            const minutes = parseInt(match[1], 10);
            const seconds = parseInt(match[2], 10);
            return minutes * 60 + seconds;
        }
        
        return null;
    }
    
    // Process and add timestamp click handlers
    function processTimestamps(element) {
        // First pass: Find timestamps in parentheses
        // Expanded regex to match more timestamp patterns:
        // - (~0:59)
        // - (~1:53, ~2:59)
        // - (~3:07)
        // - (at around 15:30)
        // - (~3:29)
        const timestampRegex = /\(~\d+:\d+(?:,\s*~\d+:\d+)*\)|\(at around \d+:\d+\)/g;
        
        // Second pass: Find standalone timestamps like "~0:59" without parentheses
        const standaloneTimestampRegex = /~\d+:\d+/g;
        
        // Third pass: Find timestamps in square brackets like [0:24] or [1:02:15]
        const squareBracketTimestampRegex = /\[(\d{1,2}:\d{2}(?::\d{2})?)\]/g;
        
        // Find all text nodes within the element
        const textNodes = [];
        const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null, false);
        
        let node;
        while (node = walker.nextNode()) {
            textNodes.push(node);
        }
        
        // Process each text node for timestamps in parentheses
        textNodes.forEach(textNode => {
            const text = textNode.nodeValue;
            if (!text) return;
            let didReplace = false;
            // Parentheses timestamps
            if (timestampRegex.test(text)) {
                timestampRegex.lastIndex = 0;
                let lastIndex = 0;
                let match;
                const fragments = [];
                while ((match = timestampRegex.exec(text)) !== null) {
                    if (match.index > lastIndex) {
                        fragments.push(document.createTextNode(text.substring(lastIndex, match.index)));
                    }
                    const fullTimestamp = match[0];
                    const individualTimestamps = fullTimestamp.match(/~\d+:\d+/g);
                    if (individualTimestamps && individualTimestamps.length > 0) {
                        const timestampLink = document.createElement('a');
                        timestampLink.href = '#';
                        timestampLink.className = 'timestamp-link text-indigo-400 hover:text-indigo-300 cursor-pointer underline';
                        timestampLink.textContent = fullTimestamp;
                        const firstTimestamp = individualTimestamps[0].substring(1);
                        const seconds = parseTimestamp(firstTimestamp);
                        timestampLink.setAttribute('data-time', seconds);
                        timestampLink.addEventListener('click', (e) => {
                            e.preventDefault();
                            seekToTime(seconds);
                        });
                        fragments.push(timestampLink);
                    } else {
                        const atAroundMatch = fullTimestamp.match(/at around (\d+:\d+)/);
                        let timestamp;
                        if (atAroundMatch) {
                            timestamp = atAroundMatch[1];
                        } else {
                            const genericMatch = fullTimestamp.match(/\d+:\d+/);
                            timestamp = genericMatch ? genericMatch[0] : null;
                        }
                        if (timestamp) {
                            const seconds = parseTimestamp(timestamp);
                            const timestampLink = document.createElement('a');
                            timestampLink.href = '#';
                            timestampLink.className = 'timestamp-link text-indigo-400 hover:text-indigo-300 cursor-pointer underline';
                            timestampLink.textContent = fullTimestamp;
                            timestampLink.setAttribute('data-time', seconds);
                            timestampLink.addEventListener('click', (e) => {
                                e.preventDefault();
                                seekToTime(seconds);
                            });
                            fragments.push(timestampLink);
                        } else {
                            fragments.push(document.createTextNode(fullTimestamp));
                        }
                    }
                    lastIndex = match.index + match[0].length;
                }
                if (lastIndex < text.length) {
                    fragments.push(document.createTextNode(text.substring(lastIndex)));
                }
                if (fragments.length > 0) {
                    const parent = textNode.parentNode;
                    fragments.forEach(fragment => {
                        parent.insertBefore(fragment, textNode);
                    });
                    parent.removeChild(textNode);
                    didReplace = true;
                }
            }
            // Standalone ~MM:SS timestamps
            if (!didReplace && standaloneTimestampRegex.test(text)) {
                standaloneTimestampRegex.lastIndex = 0;
                let lastIndex = 0;
                let match;
                const fragments = [];
                while ((match = standaloneTimestampRegex.exec(text)) !== null) {
                    if (match.index > lastIndex) {
                        fragments.push(document.createTextNode(text.substring(lastIndex, match.index)));
                    }
                    const fullTimestamp = match[0];
                    const timestamp = fullTimestamp.substring(1);
                    const seconds = parseTimestamp(timestamp);
                    const timestampLink = document.createElement('a');
                    timestampLink.href = '#';
                    timestampLink.className = 'timestamp-link text-indigo-400 hover:text-indigo-300 cursor-pointer underline';
                    timestampLink.textContent = fullTimestamp;
                    timestampLink.setAttribute('data-time', seconds);
                    timestampLink.addEventListener('click', (e) => {
                        e.preventDefault();
                        seekToTime(seconds);
                    });
                    fragments.push(timestampLink);
                    lastIndex = match.index + match[0].length;
                }
                if (lastIndex < text.length) {
                    fragments.push(document.createTextNode(text.substring(lastIndex)));
                }
                if (fragments.length > 0) {
                    const parent = textNode.parentNode;
                    fragments.forEach(fragment => {
                        parent.insertBefore(fragment, textNode);
                    });
                    parent.removeChild(textNode);
                    didReplace = true;
                }
            }
            // Square bracket [MM:SS] or [H:MM:SS] timestamps
            if (!didReplace && squareBracketTimestampRegex.test(text)) {
                squareBracketTimestampRegex.lastIndex = 0;
                let lastIndex = 0;
                let match;
                const fragments = [];
                while ((match = squareBracketTimestampRegex.exec(text)) !== null) {
                    if (match.index > lastIndex) {
                        fragments.push(document.createTextNode(text.substring(lastIndex, match.index)));
                    }
                    const fullTimestamp = match[0]; // e.g., "[0:24]"
                    const timestamp = match[1];     // e.g., "0:24" or "1:02:15"
                    // Convert to seconds
                    const parts = timestamp.split(':').map(Number);
                    let seconds = 0;
                    if (parts.length === 2) {
                        seconds = parts[0] * 60 + parts[1];
                    } else if (parts.length === 3) {
                        seconds = parts[0] * 3600 + parts[1] * 60 + parts[2];
                    }
                    const timestampLink = document.createElement('a');
                    timestampLink.href = '#';
                    timestampLink.className = 'timestamp-link text-indigo-400 hover:text-indigo-300 cursor-pointer underline';
                    timestampLink.textContent = fullTimestamp;
                    timestampLink.setAttribute('data-time', seconds);
                    timestampLink.addEventListener('click', (e) => {
                        e.preventDefault();
                        seekToTime(seconds);
                    });
                    fragments.push(timestampLink);
                    lastIndex = match.index + match[0].length;
                }
                if (lastIndex < text.length) {
                    fragments.push(document.createTextNode(text.substring(lastIndex)));
                }
                if (fragments.length > 0) {
                    const parent = textNode.parentNode;
                    fragments.forEach(fragment => {
                        parent.insertBefore(fragment, textNode);
                    });
                    parent.removeChild(textNode);
                }
            }
        });
        // Also check any existing links that might contain timestamps
        const allLinks = element.querySelectorAll('a');
        allLinks.forEach(link => {
            const text = link.textContent;
            const match = text.match(/~(\d+:\d+)/);
            if (match && !link.hasAttribute('data-time')) {
                const timestamp = match[1];
                const seconds = parseTimestamp(timestamp);
                if (seconds !== null) {
                    link.setAttribute('data-time', seconds);
                    link.className = 'timestamp-link text-indigo-400 hover:text-indigo-300 cursor-pointer underline';
                    link.href = '#';
                    link.addEventListener('click', (e) => {
                        e.preventDefault();
                        seekToTime(seconds);
                    });
                }
            }
        });
    }

    const scrollToBottom = () => {
        const conversationDiv = document.getElementById('conversation');
        conversationDiv.scrollTop = conversationDiv.scrollHeight;
    };

    /**
     * Show an error or status message in the "errorContainer" for a while.
     */
    const showError = (message, duration = 5000) => {
        if (errorContainer) {
            errorContainer.textContent = message;
            errorContainer.classList.remove('hidden');
            if (duration > 0) {
                setTimeout(() => {
                    errorContainer.classList.add('hidden');
                }, duration);
            }
        }
        console.error(message);
    };

    // Render conversation in sidebar
    const updateConversationDisplay = () => {
        const conversationDiv = document.getElementById('conversation');
        conversationDiv.innerHTML = '';
        conversationHistory.forEach(item => {
            // Use Marked to format the message content, 
            // and wrap it in a "markdown-output" or "prose" container for styling
            const markup = formatResponse(item.content);
            const messageDiv = document.createElement('div');
            messageDiv.className =
                item.role === 'assistant'
                    ? 'bg-[#223660] p-3 rounded mb-3'
                    : 'bg-blue-600 p-3 rounded mb-3';
            
            // Create a separate container for the markdown content
            const markdownContainer = document.createElement('div');
            markdownContainer.className = 'markdown-output prose prose-invert prose-ul:list-disc prose-ol:list-decimal max-w-none mt-2';
            
            // Add direct styles to ensure bullets display properly
            markdownContainer.style.cssText = 'list-style-position: outside; display: block;';
            markdownContainer.innerHTML = markup;
            
            // Fix list styling
            const lists = markdownContainer.querySelectorAll('ul, ol');
            lists.forEach(list => {
                list.style.cssText = 'list-style-position: outside !important; display: block !important; padding-left: 2.5rem !important;';
                const items = list.querySelectorAll('li');
                items.forEach(item => {
                    item.style.cssText = 'display: list-item !important; margin-bottom: 0.5rem !important; padding-left: 0 !important; text-indent: 0 !important;';
                });
            });
            
            // Process timestamps to make them clickable
            processTimestamps(markdownContainer);
            
            // Add the role label
            const roleLabel = document.createElement('strong');
            roleLabel.className = 'text-indigo-400';
            roleLabel.textContent = item.role === 'assistant' ? 'Assistant:' : 'You:';
            
            messageDiv.appendChild(roleLabel);
            messageDiv.appendChild(markdownContainer);
            conversationDiv.appendChild(messageDiv);
        });
        scrollToBottom();
    };

    // Improve summary rendering with better markdown handling
    const renderSummary = (summaryText) => {
        const parsedMarkdown = formatResponse(summaryText || 'No summary available');
        summaryContent.innerHTML = parsedMarkdown;
        
        // Apply direct styles to ensure lists render properly
        const lists = summaryContent.querySelectorAll('ul, ol');
        lists.forEach(list => {
            list.style.cssText = 'list-style-position: outside !important; display: block !important; padding-left: 2.5rem !important;';
            const items = list.querySelectorAll('li');
            items.forEach(item => {
                item.style.cssText = 'display: list-item !important; margin-bottom: 0.5rem !important; padding-left: 0 !important; text-indent: 0 !important;';
            });
        });
        
        // Process timestamps to make them clickable
        processTimestamps(summaryContent);
    };

    // API calls
    const fetchVideoData = async (url) => {
        try {
            console.log('Fetching video data for URL:', url);
            const response = await fetch('/api/video-data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ youtube_url: url }),
            });
            if (!response.ok) {
                const text = await response.text();
                throw new Error(`Server error: ${response.status} - ${text}`);
            }
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                const text = await response.text();
                throw new Error(`Expected JSON, but received: ${text}`);
            }
            const videoData = await response.json();
            console.log('Video data received:', videoData);
            return videoData;
        } catch (error) {
            console.error('Error fetching video data:', error);
            throw error;
        }
    };

    const fetchSummary = async (url) => {
        try {
            console.log('Fetching summary for URL:', url);
            const response = await fetch('/api/generate-summary', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ youtube_url: url }),
            });
            if (!response.ok) {
                const text = await response.text();
                throw new Error(`Server error: ${response.status} - ${text}`);
            }
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                const text = await response.text();
                throw new Error(`Expected JSON, but received: ${text}`);
            }
            const summaryData = await response.json();
            console.log('Summary data received:', summaryData);
            return summaryData;
        } catch (error) {
            console.error('Error fetching summary:', error);
            throw error;
        }
    };

    const fetchFollowUpResponse = async (question, transcript, title, description) => {
        try {
            console.log('Fetching follow-up response:', question);
            const response = await fetch('/api/follow-up', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question,
                    transcript,
                    title,
                    description,
                    history: conversationHistory,
                }),
            });
            if (!response.ok) {
                const text = await response.text();
                throw new Error(`Server error: ${response.status} - ${text}`);
            }
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                const text = await response.text();
                throw new Error(`Expected JSON, but received: ${text}`);
            }
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching follow-up response:', error);
            throw error;
        }
    };

    // Add fetchTranscriptViaInvidious to fetch transcript from Invidious
    async function fetchTranscriptViaInvidious(videoId) {
        const url = `https://yewtu.be/api/v1/videos/${videoId}/transcripts?format=json`;
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error('Invidious transcript fetch failed');
            const data = await response.json();
            if (!Array.isArray(data) || data.length === 0) throw new Error('No transcript data');
            // Each item: {start, duration, text}
            return data.map(item => {
                const totalSeconds = Math.floor(item.start);
                const minutes = Math.floor(totalSeconds / 60);
                const seconds = totalSeconds % 60;
                const timestamp = `[${minutes}:${seconds.toString().padStart(2, '0')}]`;
                return `${timestamp} ${item.text}`;
            }).join('\n');
        } catch (e) {
            return null;
        }
    }

    // UI States
    const setLoading = (isLoading) => {
        if (isLoading) {
            submitBtn.disabled = true;
            loadingIndicator.classList.remove('hidden');
            loadingIndicator.classList.add('flex');
            videoDataContainer.classList.add('hidden');
            videoSummary.classList.add('hidden');
            errorContainer.classList.add('hidden');
        } else {
            submitBtn.disabled = false;
            loadingIndicator.classList.add('hidden');
            loadingIndicator.classList.remove('flex');
        }
    };

    const enableFollowUp = () => {
        followUpInput.disabled = false;
        followUpSubmit.disabled = false;
        toggleSidebar.classList.remove('hidden');
        toggleSidebarDesktop.classList.remove('hidden');
    };

    const openSidebar = () => {
        sidebar.classList.remove('sidebar-closed');
        followUpInput.focus();
        adjustSidebarForMobile();
    };

    const closeSidebarFunc = () => {
        sidebar.classList.add('sidebar-closed');
        adjustSidebarForMobile();
    };

    const checkRateLimit = () => {
        const now = Date.now();
        const timeSinceLastMessage = now - lastMessageTime;
        if (timeSinceLastMessage < rateLimitCooldown && lastMessageTime !== 0) {
            followUpInput.disabled = true;
            followUpSubmit.disabled = true;
            rateLimitMessage.classList.remove('hidden');

            if (rateLimitResetTimeout) {
                clearTimeout(rateLimitResetTimeout);
            }

            const remainingTime = rateLimitCooldown - timeSinceLastMessage;
            rateLimitResetTimeout = setTimeout(() => {
                followUpInput.disabled = false;
                followUpSubmit.disabled = false;
                rateLimitMessage.classList.add('hidden');
            }, remainingTime);

            return false;
        }
        lastMessageTime = now;
        return true;
    };

    // Video Form Submission
    const handleVideoFormSubmit = async (e) => {
        e.preventDefault();
        const formData = new FormData(videoForm);
        const youtubeUrl = formData.get('youtube_url');
        if (!youtubeUrl) {
            showError('Please enter a YouTube URL');
            return;
        }
        setLoading(true);
        try {
            // Extract videoId from URL
            let videoId = null;
            const match = youtubeUrl.match(/(?:v=|youtu\.be\/|embed\/|\/v\/|\?vi=|\&v=)([\w-]{11})/);
            if (match && match[1]) {
                videoId = match[1];
            } else {
                // fallback: try to extract 11-char ID
                const idMatch = youtubeUrl.match(/([\w-]{11})/);
                if (idMatch) videoId = idMatch[1];
            }
            let transcript = null;
            if (videoId) {
                transcript = await fetchTranscriptViaInvidious(videoId);
            }
            // Fetch video data (for title, description, etc.)
            const videoData = await fetchVideoData(youtubeUrl);
            // Fetch summary, passing transcript if available
            const summaryData = await (async () => {
                const response = await fetch('/api/generate-summary', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ youtube_url: youtubeUrl, transcript }),
                });
                if (!response.ok) {
                    const text = await response.text();
                    throw new Error(`Server error: ${response.status} - ${text}`);
                }
                const contentType = response.headers.get('content-type');
                if (!contentType || !contentType.includes('application/json')) {
                    const text = await response.text();
                    throw new Error(`Expected JSON, but received: ${text}`);
                }
                return await response.json();
            })();
            // Populate
            videoTitle.textContent = videoData.title || 'No title available';
            videoDescription.textContent = videoData.description || 'No description available';
            videoTranscript.textContent = videoData.transcript || 'No transcript available';
            // Render summary as HTML with proper markdown parsing
            const summaryMarkdown = summaryData.summary || 'No summary available';
            renderSummary(summaryMarkdown);
            // Initialize YouTube player with the video ID
            initYouTubePlayer(videoData.video_id);
            videoDataContainer.classList.remove('hidden');
            playerSection.classList.remove('hidden');
            videoSummary.classList.remove('hidden');
            if (!videoSummary.contains(copySummaryBtn)) {
                videoSummary.appendChild(copySummaryBtn);
            }
            enableFollowUp();
            conversationHistory = [];
            updateConversationDisplay();
        } catch (error) {
            showError(`Failed to process video: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };

    // Follow-Up Form Submission
    const handleFollowUpFormSubmit = async (e) => {
        e.preventDefault();
        if (!checkRateLimit()) return;

        const question = followUpInput.value.trim();
        if (!question) {
            showError('Please enter a question');
            return;
        }

        const transcript = videoTranscript.textContent;
        const title = videoTitle.textContent;
        const description = videoDescription.textContent;
        if (!transcript && !title && !description) {
            showError('No video data available for follow-up questions');
            return;
        }

        try {
            // user message
            conversationHistory.push({ role: 'user', content: question });
            updateConversationDisplay();

            followUpInput.value = '';
            followUpInput.disabled = true;
            followUpSubmit.disabled = true;

            // ask server
            const response = await fetchFollowUpResponse(question, transcript, title, description);

            // assistant message
            conversationHistory.push({ role: 'assistant', content: response.response });
            updateConversationDisplay();
        } catch (error) {
            showError(`Failed to get response: ${error.message}`);
            conversationHistory.pop();
            updateConversationDisplay();
        } finally {
            followUpInput.disabled = false;
            followUpSubmit.disabled = false;
            followUpInput.focus();
        }
    };

    // Event Listeners
    videoForm.addEventListener('submit', handleVideoFormSubmit);
    followUpForm.addEventListener('submit', handleFollowUpFormSubmit);
    toggleSidebar.addEventListener('click', openSidebar);
    toggleSidebarDesktop.addEventListener('click', openSidebar);
    closeSidebar.addEventListener('click', closeSidebarFunc);

    // --- Drag Functionality ---

    // Optional: Add a subtle width transition for a smoother feel.
    sidebar.style.transition = 'width 0.2s ease';

    // Widen the handle in JS (or do it in CSS). Here we do it quickly in code:
    dragHandle.style.width = '8px'; // thicker handle for easier grabbing

    dragHandle.addEventListener('mousedown', (e) => {
        isDragging = true;
        initialX = e.clientX;
        initialWidth = sidebar.offsetWidth;
        document.addEventListener('mousemove', handleDrag);
        document.addEventListener('mouseup', handleDragEnd);
    });

    const handleDrag = (e) => {
        if (!isDragging) return;
        const deltaX = initialX - e.clientX;
        // Subtracting so dragging left = bigger width
        let newWidth = initialWidth + deltaX;

        // Keep the new width in the min / max range
        newWidth = Math.max(minSidebarWidth, Math.min(newWidth, maxSidebarWidth));
        sidebar.style.width = `${newWidth}px`;
    };

    const handleDragEnd = () => {
        isDragging = false;
        document.removeEventListener('mousemove', handleDrag);
        document.removeEventListener('mouseup', handleDragEnd);
    };

    // Mobile
    const adjustSidebarForMobile = () => {
        if (window.innerWidth <= 768) {
            sidebar.style.width = '100%';
        } else {
            // keep the previous width, or at least the minSidebarWidth
            if (parseInt(sidebar.style.width, 10) < minSidebarWidth) {
                sidebar.style.width = `${minSidebarWidth}px`;
            }
        }
    };

    window.addEventListener('resize', adjustSidebarForMobile);
    adjustSidebarForMobile();
});
