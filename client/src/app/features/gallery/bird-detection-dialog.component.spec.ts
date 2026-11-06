import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse } from '@angular/common/http';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { of, throwError } from 'rxjs';
import { ApiService } from '../../core/services/api.service';
import { BirdDetectionDialogComponent, BirdJob } from './bird-detection-dialog.component';

describe('BirdDetectionDialogComponent', () => {
  let component: any;
  let get: ReturnType<typeof vi.fn>;
  let post: ReturnType<typeof vi.fn>;
  let close: ReturnType<typeof vi.fn>;

  function job(overrides: Partial<BirdJob> = {}): BirdJob {
    return { id: 'job-1', status: 'running', phase: 'detecting', total: 1500,
      completed: 4, succeeded: 3, skipped: 1, failed: 0, cancel_requested: 0, ...overrides };
  }

  function build(paths = ['/a.jpg', '/b.jpg']) {
    get = vi.fn(() => of(job()));
    post = vi.fn(() => of(job()));
    close = vi.fn();
    TestBed.configureTestingModule({ providers: [
      { provide: ApiService, useValue: { get, post } },
      { provide: MatDialogRef, useValue: { close } },
      { provide: MAT_DIALOG_DATA, useValue: { paths } },
    ] });
    component = TestBed.runInInjectionContext(() => new BirdDetectionDialogComponent());
  }

  beforeEach(() => {
    sessionStorage.removeItem('facet_bird_detection_job');
    vi.useFakeTimers();
  });
  afterEach(() => {
    sessionStorage.removeItem('facet_bird_detection_job');
    vi.useRealTimers();
  });

  it('starts only the selected paths and persists the job for reopening', async () => {
    build();
    await component.start();
    expect(post).toHaveBeenCalledWith('/birds/jobs', {
      paths: ['/a.jpg', '/b.jpg'], tiled: true, recompute: false, min_area_percent: 1,
    });
    expect(sessionStorage.getItem('facet_bird_detection_job')).toBe('job-1');
    expect(component.job().completed).toBe(4);
    expect(component.active()).toBe(true);
  });

  it.each([0, 7])('submits the selected minimum size of %s percent', async (minimum) => {
    build();
    component.setMinArea({ target: { value: String(minimum) } });
    await component.start();
    expect(post).toHaveBeenCalledWith('/birds/jobs', {
      paths: ['/a.jpg', '/b.jpg'], tiled: true, recompute: false, min_area_percent: minimum,
    });
  });

  it('reopens an existing job without submitting another detection', async () => {
    sessionStorage.setItem('facet_bird_detection_job', 'job-1');
    build([]);
    await vi.advanceTimersByTimeAsync(0);
    expect(get).toHaveBeenCalledWith('/birds/jobs/job-1');
    expect(post).not.toHaveBeenCalled();
    expect(component.job().total).toBe(1500);
  });

  it('polls until completed and returns the exact result scope', async () => {
    build();
    await component.start();
    get.mockReturnValueOnce(of(job({ status: 'completed', phase: 'completed', completed: 1500 })));
    await vi.advanceTimersByTimeAsync(1500);
    expect(component.active()).toBe(false);
    component.finish(true);
    expect(close).toHaveBeenCalledWith({ jobId: 'job-1', changed: true, viewResults: true });
    const calls = get.mock.calls.length;
    await vi.advanceTimersByTimeAsync(6000);
    expect(get.mock.calls.length).toBe(calls);
  });

  it('requests cancellation without discarding completed results', async () => {
    build();
    await component.start();
    post.mockReturnValueOnce(of(job({ cancel_requested: 1 })));
    await component.stop();
    expect(post).toHaveBeenLastCalledWith('/birds/jobs/job-1/cancel', { confirm: true });
    expect(component.job().completed).toBe(4);
    expect(component.job().cancel_requested).toBe(1);
  });

  it('reports a busy library and permits retry', async () => {
    build();
    post.mockReturnValueOnce(throwError(() => new HttpErrorResponse({ status: 409 })));
    await component.start();
    expect(component.error()).toBe('birds.busy');
    expect(component.busy()).toBe(false);
    await component.start();
    expect(component.job().id).toBe('job-1');
  });

  it('clears an inaccessible stored job rather than polling forever', async () => {
    build();
    get.mockReturnValueOnce(throwError(() => new HttpErrorResponse({ status: 404 })));
    await component.start();
    expect(component.job()).toBeNull();
    expect(sessionStorage.getItem('facet_bird_detection_job')).toBeNull();
  });
});
