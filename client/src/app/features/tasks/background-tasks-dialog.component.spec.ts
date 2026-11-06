import { TestBed } from '@angular/core/testing';
import { Subject, of, throwError } from 'rxjs';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { BackgroundTasksDialogComponent } from './background-tasks-dialog.component';

describe('BackgroundTasksDialogComponent', () => {
  let component: any;
  let get: ReturnType<typeof vi.fn>;
  let post: ReturnType<typeof vi.fn>;
  const task = { id: 'bird-1', kind: 'birds', status: 'running', current: 3, total: 10, can_terminate: true };

  function build(response: any = of({ tasks: [task] })) {
    get = vi.fn(() => response);
    post = vi.fn(() => of({ ...task, status: 'stopping', can_terminate: false }));
    TestBed.configureTestingModule({ providers: [
      { provide: ApiService, useValue: { get, post } },
      { provide: I18nService, useValue: { t: (key: string) => key } },
    ] });
    component = TestBed.runInInjectionContext(() => new BackgroundTasksDialogComponent());
  }

  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    TestBed.resetTestingModule();
    vi.useRealTimers();
  });

  it('loads tasks once and updates progress only on manual refresh', async () => {
    build();
    await vi.advanceTimersByTimeAsync(0);
    expect(get).toHaveBeenCalledWith('/tasks');
    expect(component.activeCount()).toBe(1);
    expect(component.percent(component.tasks()[0])).toBe(30);
    get.mockReturnValue(of({ tasks: [{ ...task, status: 'completed', current: 10 }] }));
    await vi.advanceTimersByTimeAsync(10000);
    expect(get).toHaveBeenCalledTimes(1);
    expect(component.tasks()[0].status).toBe('running');
    await component.refresh();
    expect(component.tasks()[0].status).toBe('completed');
    expect(component.activeCount()).toBe(0);
    // 已完成任务之后也只在手动刷新时获取新任务。
    get.mockReturnValue(of({ tasks: [task] }));
    await vi.advanceTimersByTimeAsync(10000);
    expect(get).toHaveBeenCalledTimes(2);
    expect(component.activeCount()).toBe(0);
    await component.refresh();
    expect(component.activeCount()).toBe(1);
  });

  it('preserves tasks after errors and retries only on manual refresh', async () => {
    build();
    await vi.advanceTimersByTimeAsync(0);
    get.mockReturnValueOnce(throwError(() => new Error('offline')));
    await component.refresh();
    expect(component.error()).toBe(true);
    expect(component.tasks()).toEqual([task]);
    await vi.advanceTimersByTimeAsync(10000);
    expect(get).toHaveBeenCalledTimes(2);
    expect(component.error()).toBe(true);
    await component.refresh();
    expect(component.error()).toBe(false);
  });

  it('does not overlap requests or update after closing', async () => {
    const pending = new Subject<any>();
    build(pending);
    await component.refresh();
    expect(get).toHaveBeenCalledTimes(1);
    TestBed.resetTestingModule();
    pending.next({ tasks: [task] });
    pending.complete();
    await vi.advanceTimersByTimeAsync(10000);
    expect(get).toHaveBeenCalledTimes(1);
    expect(component.tasks()).toEqual([]);
  });

  it('ignores manual refresh after the dialog is destroyed', async () => {
    build();
    await vi.advanceTimersByTimeAsync(0);
    TestBed.resetTestingModule();
    await component.refresh();
    await vi.advanceTimersByTimeAsync(10000);
    expect(get).toHaveBeenCalledTimes(1);
  });

  it('requests termination and keeps the returned progress without automatic polling', async () => {
    build();
    await vi.advanceTimersByTimeAsync(0);
    await component.terminate(component.tasks()[0]);
    expect(post).toHaveBeenCalledWith('/tasks/bird-1/terminate', { confirm: true });
    expect(component.tasks()[0].status).toBe('stopping');
    expect(component.tasks()[0].current).toBe(3);
    await component.terminate(component.tasks()[0]);
    await vi.advanceTimersByTimeAsync(10000);
    expect(post).toHaveBeenCalledTimes(1);
    expect(get).toHaveBeenCalledTimes(1);
  });

  it('prevents overlapping mutations and refresh while termination is pending', async () => {
    build();
    await vi.advanceTimersByTimeAsync(0);
    const pending = new Subject<any>();
    post.mockReturnValue(pending);
    const action = component.terminate(task);
    await component.terminate(task);
    await component.refresh();
    expect(post).toHaveBeenCalledTimes(1);
    expect(get).toHaveBeenCalledTimes(1);
    pending.next({ ...task, status: 'stopping', can_terminate: false });
    pending.complete();
    await action;
    expect(component.terminating()).toBeNull();
  });

  it('retains task state after termination failure and allows retry', async () => {
    build();
    await vi.advanceTimersByTimeAsync(0);
    post.mockReturnValueOnce(throwError(() => new Error('offline')));
    await component.terminate(task);
    expect(component.actionError()).toBe('tasks.terminate_error');
    expect(component.tasks()).toEqual([task]);
    await component.terminate(task);
    expect(component.actionError()).toBe('');
    expect(component.tasks()[0].status).toBe('stopping');
  });

  it('does not update a closed dialog after a pending termination request', async () => {
    build();
    await vi.advanceTimersByTimeAsync(0);
    const pending = new Subject<any>();
    post.mockReturnValue(pending);
    const action = component.terminate(task);
    TestBed.resetTestingModule();
    pending.next({ ...task, status: 'stopping' });
    pending.complete();
    await action;
    expect(component.tasks()).toEqual([task]);
    await component.terminate(task);
    expect(post).toHaveBeenCalledTimes(1);
  });
});
