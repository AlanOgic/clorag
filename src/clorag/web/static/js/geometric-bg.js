/**
 * Animated Geometric Background with CYANVIEW Letters
 * Configurable canvas animation with floating shapes and letter formations
 */

class GeometricBackground {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.error(`Canvas element with id "${canvasId}" not found`);
            return;
        }
        this.ctx = this.canvas.getContext('2d');

        // Merge options with defaults
        this.config = {
            shapeCount: options.shapeCount ?? 25,
            connectionDistance: options.connectionDistance ?? 150,
            speed: { min: 0.15, max: 0.4 },
            rotationSpeed: { min: 0.001, max: 0.003 },
            sizeRange: { min: 15, max: 55 },
            letterSizeRange: { min: 35, max: 70 },
            letterMultiplier: options.letterMultiplier ?? 4,
            colors: {
                cyan: { r: 0, g: 168, b: 232 },
                white: { r: 255, g: 255, b: 255 }
            },
            mouseAttraction: {
                radius: 200,
                strength: 0.02,
                maxSpeed: 2
            },
            formation: {
                interval: options.formationInterval ?? 90000,
                holdDuration: options.formationHoldDuration ?? 5000,
                formingDuration: options.formationFormingDuration ?? 5000,
                disperseDuration: options.formationDisperseDuration ?? 2000,
                letterSpacing: 55,
                firstDelay: options.formationFirstDelay ?? 10000,
                yPosition: options.formationYPosition ?? 0.2 // 0-1, percentage from top
            },
            background: options.background ?? {
                color1: '#003366',
                color2: '#0077b6'
            },
            drawBackground: options.drawBackground ?? true
        };

        this.CYANVIEW_LETTERS = ['C', 'Y', 'A', 'N', 'V', 'I', 'E', 'W'];
        this.shapes = [];
        this.letters = [];
        this.animationId = null;
        this.mouse = { x: null, y: null, active: false };
        this.lastFormationTime = 0;

        // Formation state
        this.formation = {
            active: false,
            activeSet: -1,
            startTime: 0,
            phase: 'idle'
        };

        this.SHAPE_TYPES = ['hexagon', 'pentagon', 'triangle', 'circle'];

        this.init();
    }

    // Easing functions
    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    easeOutQuad(t) {
        return 1 - (1 - t) * (1 - t);
    }

    // Shape class
    createShape() {
        const shape = {
            type: 'shape',
            x: Math.random() * this.canvas.width,
            y: Math.random() * this.canvas.height,
            size: this.config.sizeRange.min + Math.random() * (this.config.sizeRange.max - this.config.sizeRange.min),
            shapeType: this.SHAPE_TYPES[Math.floor(Math.random() * this.SHAPE_TYPES.length)],
            rotation: Math.random() * Math.PI * 2,
            rotationSpeed: (this.config.rotationSpeed.min + Math.random() * (this.config.rotationSpeed.max - this.config.rotationSpeed.min)) * (Math.random() > 0.5 ? 1 : -1),
            vx: 0,
            vy: 0,
            isCyan: Math.random() > 0.3,
            baseOpacity: 0.1 + Math.random() * 0.2,
            pulseOffset: Math.random() * Math.PI * 2,
            pulseSpeed: 0.002 + Math.random() * 0.004,
            currentOpacity: 0.15
        };

        const speed = this.config.speed.min + Math.random() * (this.config.speed.max - this.config.speed.min);
        const angle = Math.random() * Math.PI * 2;
        shape.vx = Math.cos(angle) * speed;
        shape.vy = Math.sin(angle) * speed;

        return shape;
    }

    // Letter class
    createLetter(char) {
        const letter = {
            type: 'letter',
            char: char,
            x: Math.random() * this.canvas.width,
            y: Math.random() * this.canvas.height,
            size: this.config.letterSizeRange.min + Math.random() * (this.config.letterSizeRange.max - this.config.letterSizeRange.min),
            rotation: Math.random() * Math.PI * 2,
            rotationSpeed: (this.config.rotationSpeed.min + Math.random() * (this.config.rotationSpeed.max - this.config.rotationSpeed.min)) * 0.3 * (Math.random() > 0.5 ? 1 : -1),
            vx: 0,
            vy: 0,
            isCyan: Math.random() > 0.2,
            baseOpacity: 0.15 + Math.random() * 0.25,
            pulseOffset: Math.random() * Math.PI * 2,
            pulseSpeed: 0.001 + Math.random() * 0.002,
            scaleOffset: Math.random() * Math.PI * 2,
            scaleSpeed: 0.0015 + Math.random() * 0.002,
            scaleAmount: 0.1 + Math.random() * 0.1,
            currentOpacity: 0.2,
            currentScale: 1,
            inFormation: false,
            formationTarget: null,
            formationOpacityBoost: 1
        };

        const speed = this.config.speed.min + Math.random() * (this.config.speed.max - this.config.speed.min);
        const angle = Math.random() * Math.PI * 2;
        letter.vx = Math.cos(angle) * speed;
        letter.vy = Math.sin(angle) * speed;

        return letter;
    }

    updateShape(shape, time) {
        shape.x += shape.vx;
        shape.y += shape.vy;
        shape.rotation += shape.rotationSpeed;

        const padding = shape.size;
        if (shape.x < -padding) shape.x = this.canvas.width + padding;
        if (shape.x > this.canvas.width + padding) shape.x = -padding;
        if (shape.y < -padding) shape.y = this.canvas.height + padding;
        if (shape.y > this.canvas.height + padding) shape.y = -padding;

        shape.currentOpacity = shape.baseOpacity + Math.sin(time * shape.pulseSpeed + shape.pulseOffset) * 0.1;
    }

    updateLetter(letter, time) {
        // Skip mouse attraction during formation
        if (!letter.inFormation) {
            if (this.mouse.active && this.mouse.x !== null) {
                const dx = this.mouse.x - letter.x;
                const dy = this.mouse.y - letter.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < this.config.mouseAttraction.radius && distance > 0) {
                    const force = (1 - distance / this.config.mouseAttraction.radius) * this.config.mouseAttraction.strength;
                    letter.vx += (dx / distance) * force;
                    letter.vy += (dy / distance) * force;

                    const speed = Math.sqrt(letter.vx * letter.vx + letter.vy * letter.vy);
                    if (speed > this.config.mouseAttraction.maxSpeed) {
                        letter.vx = (letter.vx / speed) * this.config.mouseAttraction.maxSpeed;
                        letter.vy = (letter.vy / speed) * this.config.mouseAttraction.maxSpeed;
                    }
                }
            } else {
                const speed = Math.sqrt(letter.vx * letter.vx + letter.vy * letter.vy);
                const targetSpeed = this.config.speed.min + Math.random() * (this.config.speed.max - this.config.speed.min);
                if (speed > targetSpeed * 1.5) {
                    letter.vx *= 0.98;
                    letter.vy *= 0.98;
                }
            }
        }

        letter.x += letter.vx;
        letter.y += letter.vy;
        letter.rotation += letter.rotationSpeed;

        const padding = letter.size;
        if (letter.x < -padding) letter.x = this.canvas.width + padding;
        if (letter.x > this.canvas.width + padding) letter.x = -padding;
        if (letter.y < -padding) letter.y = this.canvas.height + padding;
        if (letter.y > this.canvas.height + padding) letter.y = -padding;

        letter.currentOpacity = letter.baseOpacity + Math.sin(time * letter.pulseSpeed + letter.pulseOffset) * 0.1;
        letter.currentScale = 1 + Math.sin(time * letter.scaleSpeed + letter.scaleOffset) * letter.scaleAmount;
    }

    drawShape(shape) {
        const color = shape.isCyan ? this.config.colors.cyan : this.config.colors.white;

        this.ctx.save();
        this.ctx.translate(shape.x, shape.y);
        this.ctx.rotate(shape.rotation);

        this.ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${shape.currentOpacity * 0.5})`;
        this.ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${shape.currentOpacity + 0.2})`;
        this.ctx.lineWidth = 1.5;

        this.ctx.beginPath();

        if (shape.shapeType === 'circle') {
            this.ctx.arc(0, 0, shape.size / 2, 0, Math.PI * 2);
        } else {
            const sides = shape.shapeType === 'triangle' ? 3 : shape.shapeType === 'pentagon' ? 5 : 6;
            const radius = shape.size / 2;
            for (let i = 0; i < sides; i++) {
                const angle = (i / sides) * Math.PI * 2 - Math.PI / 2;
                const x = Math.cos(angle) * radius;
                const y = Math.sin(angle) * radius;
                if (i === 0) this.ctx.moveTo(x, y);
                else this.ctx.lineTo(x, y);
            }
        }

        this.ctx.closePath();
        this.ctx.fill();
        this.ctx.stroke();
        this.ctx.restore();
    }

    drawLetter(letter) {
        const boostLevel = (letter.formationOpacityBoost - 1) / 1.5;
        const color = boostLevel > 0.5 ? this.config.colors.cyan :
                      (letter.isCyan ? this.config.colors.cyan : this.config.colors.white);

        this.ctx.save();
        this.ctx.translate(letter.x, letter.y);
        this.ctx.rotate(letter.rotation);
        this.ctx.scale(letter.currentScale, letter.currentScale);

        this.ctx.font = `700 ${letter.size}px 'Phenomena', 'Segoe UI', Arial, sans-serif`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';

        const fillOpacity = Math.min(letter.currentOpacity * 0.6 * letter.formationOpacityBoost, 0.9);
        const strokeOpacity = Math.min((letter.currentOpacity + 0.25) * letter.formationOpacityBoost, 1);

        this.ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${fillOpacity})`;
        this.ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${strokeOpacity})`;
        this.ctx.lineWidth = 1.5 + boostLevel * 0.5;

        this.ctx.fillText(letter.char, 0, 0);
        this.ctx.strokeText(letter.char, 0, 0);

        this.ctx.restore();
    }

    drawConnections() {
        const allElements = [...this.shapes, ...this.letters];
        this.ctx.lineWidth = 0.5;

        for (let i = 0; i < allElements.length; i++) {
            for (let j = i + 1; j < allElements.length; j++) {
                const dx = allElements[i].x - allElements[j].x;
                const dy = allElements[i].y - allElements[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < this.config.connectionDistance) {
                    const opacity = (1 - distance / this.config.connectionDistance) * 0.15;
                    this.ctx.strokeStyle = `rgba(0, 168, 232, ${opacity})`;
                    this.ctx.beginPath();
                    this.ctx.moveTo(allElements[i].x, allElements[i].y);
                    this.ctx.lineTo(allElements[j].x, allElements[j].y);
                    this.ctx.stroke();
                }
            }
        }
    }

    drawBackground() {
        if (!this.config.drawBackground) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            return;
        }

        const gradient = this.ctx.createLinearGradient(0, 0, this.canvas.width, this.canvas.height);
        gradient.addColorStop(0, this.config.background.color1);
        gradient.addColorStop(1, this.config.background.color2);
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    // Formation methods
    getFormationTargets() {
        const totalWidth = (this.CYANVIEW_LETTERS.length - 1) * this.config.formation.letterSpacing;
        const startX = (this.canvas.width - totalWidth) / 2;
        const targetY = this.canvas.height * this.config.formation.yPosition;

        return this.CYANVIEW_LETTERS.map((_, i) => ({
            x: startX + i * this.config.formation.letterSpacing,
            y: targetY
        }));
    }

    startFormation(time) {
        this.formation.active = true;
        this.formation.activeSet = Math.floor(Math.random() * this.config.letterMultiplier);
        this.formation.startTime = time;
        this.formation.phase = 'forming';

        const targets = this.getFormationTargets();
        const setStart = this.formation.activeSet * 8;

        for (let i = 0; i < 8; i++) {
            const letter = this.letters[setStart + i];
            letter.formationTarget = targets[i];
            letter.formationStartX = letter.x;
            letter.formationStartY = letter.y;
            letter.formationStartRotation = letter.rotation;
            letter.inFormation = true;
            letter.vx = 0;
            letter.vy = 0;
        }
    }

    updateFormation(time) {
        if (!this.formation.active) return;

        const elapsed = time - this.formation.startTime;
        const setStart = this.formation.activeSet * 8;

        if (this.formation.phase === 'forming') {
            const progress = Math.min(elapsed / this.config.formation.formingDuration, 1);
            const easedProgress = this.easeInOutCubic(progress);

            for (let i = 0; i < 8; i++) {
                const letter = this.letters[setStart + i];
                const target = letter.formationTarget;

                letter.x = letter.formationStartX + (target.x - letter.formationStartX) * easedProgress;
                letter.y = letter.formationStartY + (target.y - letter.formationStartY) * easedProgress;
                letter.rotation = letter.formationStartRotation * (1 - easedProgress);
                letter.formationOpacityBoost = 1 + easedProgress * 1.5;
            }

            if (progress >= 1) {
                this.formation.phase = 'holding';
                this.formation.startTime = time;
                for (let i = 0; i < 8; i++) {
                    const letter = this.letters[setStart + i];
                    letter.x = letter.formationTarget.x;
                    letter.y = letter.formationTarget.y;
                    letter.rotation = 0;
                    letter.formationOpacityBoost = 2.5;
                }
            }
        } else if (this.formation.phase === 'holding') {
            for (let i = 0; i < 8; i++) {
                const letter = this.letters[setStart + i];
                letter.x = letter.formationTarget.x + Math.sin(time * 0.001 + i * 0.5) * 3;
                letter.y = letter.formationTarget.y + Math.cos(time * 0.0015 + i * 0.5) * 2;
                letter.rotation = Math.sin(time * 0.0008 + i) * 0.03;
                letter.formationOpacityBoost = 2.5;
            }

            if (elapsed > this.config.formation.holdDuration) {
                this.formation.phase = 'dispersing';
                this.formation.startTime = time;
                for (let i = 0; i < 8; i++) {
                    const letter = this.letters[setStart + i];
                    letter.disperseStartX = letter.x;
                    letter.disperseStartY = letter.y;
                    letter.disperseTargetX = Math.random() * this.canvas.width;
                    letter.disperseTargetY = Math.random() * this.canvas.height;
                    const angle = Math.random() * Math.PI * 2;
                    const speed = this.config.speed.min + Math.random() * (this.config.speed.max - this.config.speed.min);
                    letter.disperseVx = Math.cos(angle) * speed;
                    letter.disperseVy = Math.sin(angle) * speed;
                }
            }
        } else if (this.formation.phase === 'dispersing') {
            const progress = Math.min(elapsed / this.config.formation.disperseDuration, 1);
            const easedProgress = this.easeOutQuad(progress);

            for (let i = 0; i < 8; i++) {
                const letter = this.letters[setStart + i];
                letter.vx = letter.disperseVx * easedProgress;
                letter.vy = letter.disperseVy * easedProgress;
                letter.formationOpacityBoost = 2.5 - easedProgress * 1.5;
            }

            if (progress >= 1) {
                this.formation.active = false;
                this.formation.phase = 'idle';
                for (let i = 0; i < 8; i++) {
                    const letter = this.letters[setStart + i];
                    letter.inFormation = false;
                    letter.formationOpacityBoost = 1;
                    letter.vx = letter.disperseVx;
                    letter.vy = letter.disperseVy;
                }
            }
        }
    }

    animate(time) {
        this.drawBackground();

        // Check formation timing
        if (!this.formation.active) {
            const timeSinceLastFormation = time - this.lastFormationTime;
            const delay = this.lastFormationTime === 0 ? this.config.formation.firstDelay : this.config.formation.interval;
            if (timeSinceLastFormation > delay) {
                this.startFormation(time);
                this.lastFormationTime = time;
            }
        }

        this.updateFormation(time);

        // Update and draw shapes
        this.shapes.forEach(shape => {
            this.updateShape(shape, time);
            this.drawShape(shape);
        });

        // Update and draw letters
        this.letters.forEach(letter => {
            this.updateLetter(letter, time);
            this.drawLetter(letter);
        });

        this.drawConnections();

        this.animationId = requestAnimationFrame((t) => this.animate(t));
    }

    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    init() {
        this.resize();

        // Create shapes
        this.shapes = [];
        for (let i = 0; i < this.config.shapeCount; i++) {
            this.shapes.push(this.createShape());
        }

        // Create letters
        this.letters = [];
        for (let i = 0; i < this.config.letterMultiplier; i++) {
            this.CYANVIEW_LETTERS.forEach(char => {
                this.letters.push(this.createLetter(char));
            });
        }

        // Event listeners
        window.addEventListener('resize', () => this.resize());

        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouse.x = e.clientX - rect.left;
            this.mouse.y = e.clientY - rect.top;
            this.mouse.active = true;
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.mouse.active = false;
        });

        // Start animation
        this.animate(0);
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
}

// Export for use
window.GeometricBackground = GeometricBackground;
