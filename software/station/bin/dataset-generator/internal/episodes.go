package internal

import (
	"fmt"
	"norma_core/target/generated-sources/protobuf/drivers/inferences/normvla"
	"os"
	"path/filepath"
	"time"

	"github.com/norma-core/normfs/normfs_go/uintn"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/parquet-go/parquet-go"
	"github.com/rs/zerolog/log"
)

type EpisodeDetector struct {
	path        string
	duration    time.Duration
	task        string
	minCommands uint64

	lastGoals         []uint32
	commandsCnt       uint64
	lastMovementStamp uint64

	episodeIndex          uint64
	episodeMonotonicStart uint64
	episodeLocalStart     uint64
	episodeSkipReason     string
	episodeSkipFrameId    string

	buffer []Frame

	stats   Stats
	program *tea.Program
}

type Stats struct {
	EpisodesStarted              uint64
	EpisodesSaved                uint64
	EpisodesDiscarded            uint64
	TotalFramesSaved             uint64
	FramesSkippedWrongImageCount uint64
}

type EpisodeDetectorConfig struct {
	Path        string
	Duration    time.Duration
	Task        string
	MinCommands uint64
	Program     *tea.Program
}

func NewEpisodeDetector(cfg EpisodeDetectorConfig) *EpisodeDetector {
	return &EpisodeDetector{
		path:        cfg.Path,
		duration:    cfg.Duration,
		task:        cfg.Task,
		minCommands: cfg.MinCommands,
		program:     cfg.Program,
	}
}

func (ed *EpisodeDetector) ProcessFrame(frame *normvla.FrameReader) {
	// Skip frames without any images
	if len(frame.GetImages()) == 0 {
		ed.stats.FramesSkippedWrongImageCount++

		// Log first occurrence with details
		if ed.stats.FramesSkippedWrongImageCount == 1 {
			frameId, _ := uintn.FromLEBytes(frame.GetGlobalFrameId())
			log.Warn().Msgf("⚠️  Skipping frames with no images (first at frame %v)", frameId)
		}

		// Log periodic progress every 100 skipped frames
		if ed.stats.FramesSkippedWrongImageCount%100 == 0 {
			log.Warn().Msgf("⚠️  Skipped %d frames due to no images",
				ed.stats.FramesSkippedWrongImageCount)
		}

		return
	}

	stamp := frame.GetMonotonicStampNs()
	joints := frame.GetJoints()

	// Extract current goals
	currentGoals := make([]uint32, len(joints))
	for i, j := range joints {
		currentGoals[i] = j.GetGoal()
	}

	commandsChanged := ed.hasCommandsChanged(currentGoals)

	// Not in episode - check if we should start one
	if ed.episodeMonotonicStart == 0 {
		if commandsChanged {
			ed.startNewEpisode(frame)
		}
		ed.updateGoalTracking(currentGoals)
		return
	}

	// In episode - track commands
	if commandsChanged {
		ed.commandsCnt++
		ed.updateGoalTracking(currentGoals)
		ed.lastMovementStamp = stamp
	}

	inEpisodeStamp := stamp - ed.episodeMonotonicStart

	// Check frame gap - mark episode for skip if gap > 500ms
	if ed.episodeSkipReason == "" && len(ed.buffer) > 0 {
		lastStamp := ed.buffer[len(ed.buffer)-1].TimestampNsSinceEpisodeStart
		gap := inEpisodeStamp - lastStamp
		if gap > uint64(500*time.Millisecond) {
			ed.markEpisodeSkip(frame, fmt.Sprintf("frame gap %dms > 500ms", gap/uint64(time.Millisecond)))
		}
	}

	// Buffer frame only if episode not marked for skip
	if ed.episodeSkipReason == "" {
		ed.bufferFrame(frame, inEpisodeStamp)
	}

	// Check if episode should end
	if time.Duration(inEpisodeStamp) >= ed.duration &&
		time.Duration(stamp-ed.lastMovementStamp) >= 5*time.Second {
		ed.trySaveCurrentEpisode()
		ed.resetEpisode()
	}
}

func (ed *EpisodeDetector) markEpisodeSkip(frame *normvla.FrameReader, reason string) {
	if ed.episodeSkipReason != "" {
		return
	}
	ed.episodeSkipReason = reason
	frameId, _ := uintn.FromLEBytes(frame.GetGlobalFrameId())
	ed.episodeSkipFrameId = frameId.String()
	log.Warn().Msgf("Episode #%d marked for skip at frame %v: %s", ed.episodeIndex, frameId, reason)
	if ed.program != nil {
		ed.program.Send(EpisodeSkipMarkedMsg{Index: ed.episodeIndex, Reason: reason})
	}
}

func (ed *EpisodeDetector) startNewEpisode(frame *normvla.FrameReader) {
	stamp := frame.GetMonotonicStampNs()

	ed.episodeMonotonicStart = stamp
	ed.episodeLocalStart = uint64(time.Now().UnixNano()) // approximate
	ed.lastMovementStamp = stamp
	ed.commandsCnt = 1
	ed.stats.EpisodesStarted++

	// Buffer first frame
	ed.buffer = []Frame{}
	ed.bufferFrame(frame, 0)

	frameId, _ := uintn.FromLEBytes(frame.GetGlobalFrameId())
	log.Info().Msgf("Started episode #%d at frame %v", ed.episodeIndex, frameId)
	if ed.program != nil {
		ed.program.Send(EpisodeStartedMsg{Index: ed.episodeIndex})
	}
}

func (ed *EpisodeDetector) bufferFrame(frame *normvla.FrameReader, inEpisodeStamp uint64) {
	f := Frame{
		EpisodeStartNS:               ed.episodeLocalStart,
		GlobalFrameID:                frame.GetGlobalFrameId(),
		TimestampNsSinceEpisodeStart: inEpisodeStamp,
		Task:                         ed.task,
		Joints:                       convertJoints(frame.GetJoints()),
		Images:                       convertImages(frame.GetImages()),
	}
	ed.buffer = append(ed.buffer, f)

	if ed.program != nil {
		ed.program.Send(FrameBufferedMsg{})
	}
}

func (ed *EpisodeDetector) trySaveCurrentEpisode() {
	if ed.episodeSkipReason != "" {
		log.Warn().Msgf("Episode #%d discarded: %s", ed.episodeIndex, ed.episodeSkipReason)
		if ed.program != nil {
			ed.program.Send(EpisodeDiscardedMsg{Index: ed.episodeIndex, Reason: ed.episodeSkipReason, FrameId: ed.episodeSkipFrameId})
		}
		ed.stats.EpisodesDiscarded++
		return
	}

	if ed.commandsCnt < ed.minCommands {
		reason := fmt.Sprintf("insufficient commands (%d < %d)", ed.commandsCnt, ed.minCommands)
		firstId, _ := uintn.FromLEBytes(ed.buffer[0].GlobalFrameID)
		lastId, _ := uintn.FromLEBytes(ed.buffer[len(ed.buffer)-1].GlobalFrameID)
		log.Warn().Msgf("Episode #%d discarded [%v -> %v]: %s",
			ed.episodeIndex, firstId, lastId, reason)
		if ed.program != nil {
			lastFrame := ed.buffer[len(ed.buffer)-1]
			frameId, _ := uintn.FromLEBytes(lastFrame.GlobalFrameID)
			ed.program.Send(EpisodeDiscardedMsg{Index: ed.episodeIndex, Reason: reason, FrameId: frameId.String()})
		}
		ed.stats.EpisodesDiscarded++
		return
	}

	if len(ed.buffer) == 0 {
		log.Warn().Msgf("Episode #%d discarded: no frames", ed.episodeIndex)
		if ed.program != nil {
			lastFrame := ed.buffer[len(ed.buffer)-1]
			frameId, _ := uintn.FromLEBytes(lastFrame.GlobalFrameID)
			ed.program.Send(EpisodeDiscardedMsg{Index: ed.episodeIndex, Reason: "no frames", FrameId: frameId.String()})
		}
		ed.stats.EpisodesDiscarded++
		return
	}

	// Save to parquet
	filePath := filepath.Join(ed.path, fmt.Sprintf("%d.parquet", ed.episodeIndex))
	if err := os.MkdirAll(ed.path, 0755); err != nil {
		log.Error().Err(err).Msgf("Failed to create output directory %s", ed.path)
		return
	}

	outFile, err := os.Create(filePath)
	if err != nil {
		log.Error().Err(err).Msgf("Failed to create parquet file %s", filePath)
		return
	}
	defer outFile.Close()

	if err := parquet.Write(outFile, ed.buffer); err != nil {
		log.Error().Err(err).Msgf("Failed to write parquet file %s", filePath)
		return
	}

	firstId, _ := uintn.FromLEBytes(ed.buffer[0].GlobalFrameID)
	lastId, _ := uintn.FromLEBytes(ed.buffer[len(ed.buffer)-1].GlobalFrameID)
	log.Info().Msgf("Episode #%d saved [%v -> %v] to %s with %d frames", ed.episodeIndex, firstId, lastId, filePath, len(ed.buffer))
	if ed.program != nil {
		ed.program.Send(EpisodeSavedMsg{Index: ed.episodeIndex, Frames: len(ed.buffer)})
	}

	ed.stats.EpisodesSaved++
	ed.stats.TotalFramesSaved += uint64(len(ed.buffer))
	ed.episodeIndex++
}

func (ed *EpisodeDetector) resetEpisode() {
	ed.episodeMonotonicStart = 0
	ed.episodeLocalStart = 0
	ed.episodeSkipReason = ""
	ed.episodeSkipFrameId = ""
	ed.commandsCnt = 0
	ed.buffer = nil
}

func (ed *EpisodeDetector) hasCommandsChanged(currentGoals []uint32) bool {
	if len(ed.lastGoals) == 0 {
		return false
	}
	if len(ed.lastGoals) != len(currentGoals) {
		return true
	}
	for i := range currentGoals {
		if ed.lastGoals[i] != currentGoals[i] {
			return true
		}
	}
	return false
}

func (ed *EpisodeDetector) updateGoalTracking(goals []uint32) {
	ed.lastGoals = make([]uint32, len(goals))
	copy(ed.lastGoals, goals)
}

func (ed *EpisodeDetector) GetSkipInfo() (count uint64, reason string) {
	if ed.stats.FramesSkippedWrongImageCount > 0 {
		return ed.stats.FramesSkippedWrongImageCount, "no images"
	}
	return 0, ""
}

func (ed *EpisodeDetector) Finalize() {
	if ed.episodeMonotonicStart != 0 {
		ed.trySaveCurrentEpisode()
	}
	ed.mergeParquetFiles()
	ed.printStats()
}

func (ed *EpisodeDetector) mergeParquetFiles() {
	if ed.episodeIndex == 0 {
		log.Info().Msg("No parquet files to merge")
		return
	}

	var files []*os.File
	var rowGroups []parquet.RowGroup
	defer func() {
		for _, f := range files {
			f.Close()
		}
	}()

	for i := uint64(0); i < ed.episodeIndex; i++ {
		path := filepath.Join(ed.path, fmt.Sprintf("%d.parquet", i))
		f, err := os.Open(path)
		if err != nil {
			log.Error().Err(err).Msgf("Failed to open parquet file %s", path)
			return
		}
		files = append(files, f)

		stat, _ := f.Stat()
		pf, err := parquet.OpenFile(f, stat.Size())
		if err != nil {
			log.Error().Err(err).Msgf("Failed to read parquet file %s", path)
			return
		}
		rowGroups = append(rowGroups, pf.RowGroups()...)
	}

	outputName := filepath.Base(ed.path) + ".parquet"
	outputPath := filepath.Join(filepath.Dir(ed.path), outputName)

	outFile, err := os.Create(outputPath)
	if err != nil {
		log.Error().Err(err).Msgf("Failed to create merged parquet file %s", outputPath)
		return
	}
	defer outFile.Close()

	writer := parquet.NewGenericWriter[Frame](outFile)
	for _, rg := range rowGroups {
		if _, err := parquet.CopyRows(writer, rg.Rows()); err != nil {
			log.Error().Err(err).Msg("Failed to copy rows")
			return
		}
	}
	if err := writer.Close(); err != nil {
		log.Error().Err(err).Msg("Failed to close writer")
		return
	}

	// Cleanup individual files
	for i := uint64(0); i < ed.episodeIndex; i++ {
		f := filepath.Join(ed.path, fmt.Sprintf("%d.parquet", i))
		os.Remove(f)
	}
	os.Remove(ed.path)

	var totalRows int64
	for _, rg := range rowGroups {
		totalRows += rg.NumRows()
	}
	log.Info().Msgf("Merged %d episodes into %s with %d total frames", ed.episodeIndex, outputPath, totalRows)
}

func (ed *EpisodeDetector) printStats() {
	log.Info().Msg("=== Statistics ===")
	log.Info().Msgf("Episodes started:   %d", ed.stats.EpisodesStarted)
	log.Info().Msgf("Episodes saved:     %d", ed.stats.EpisodesSaved)
	log.Info().Msgf("Episodes discarded: %d", ed.stats.EpisodesDiscarded)
	log.Info().Msgf("Total frames saved: %d", ed.stats.TotalFramesSaved)

	if ed.stats.FramesSkippedWrongImageCount > 0 {
		log.Warn().Msgf("⚠️  Frames skipped: %d (no images)",
			ed.stats.FramesSkippedWrongImageCount)
	}
}

func convertJoints(joints []*normvla.JointReader) []Joint {
	res := make([]Joint, len(joints))
	for i, j := range joints {
		res[i] = Joint{
			RangeMin:     j.GetRangeMin(),
			RangeMax:     j.GetRangeMax(),
			Position:     j.GetPosition(),
			PositionNorm: j.GetPositionNorm(),
			Goal:         j.GetGoal(),
			GoalNorm:     j.GetGoalNorm(),
			CurrentMA:    j.GetCurrentMa(),
			Velocity:     j.GetVelocity(),
		}
	}
	return res
}

func convertImages(images []*normvla.ImageReader) []Image {
	res := make([]Image, len(images))
	for i, img := range images {
		res[i] = Image{
			JPEG: img.GetJpeg(),
		}
	}
	return res
}
