import { TestBed } from '@angular/core/testing';
import { Subject, of, throwError } from 'rxjs';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { BackgroundTasksDialogComponent } from './background-tasks-dialog.component';

describe('BackgroundTasksDialogComponent', () => {
  let component: any;
  let get: ReturnType<typeof vi.fn>;
  const task = { id: 'bird-1', kind: 'birds', status: 'running', current: 3, total: 10 };

  function build(response: any = of({ tasks: [task] })) {
    get = vi.fn(() => response);
    TestBed.configureTestingModule({ providers: [
      { provide: ApiService, useValue: { get } },
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
});
